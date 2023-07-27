mod args;
mod geojson;
mod utils;
mod features;
use std::{fs::File, io::{BufReader, BufWriter}, sync::{Arc, Mutex, atomic::{AtomicU32}}, process::exit};
use log::{error, debug, info};
use args::{ARGS, Args};
use polars::prelude::*;
use rayon::prelude::*;
use tch::Tensor;

#[derive(Clone)]
enum InputImage {
    Slide(Arc<Mutex<openslide_rs::OpenSlide>>),
    Image(Arc<Mutex<Tensor>>),
}

fn load_input_image()-> InputImage {
    match ARGS.slide.extension().and_then(|ext|ext.to_str()) {
        Some("svs") => {
            InputImage::Slide(Arc::new(Mutex::new(openslide_rs::OpenSlide::new(&ARGS.slide).unwrap())))
        },
        Some("png") | Some("jpg") | Some("jpeg") => {
            InputImage::Image(Arc::new(Mutex::new(tch::vision::image::load(&ARGS.slide).unwrap())))
        },
        _ => {
            error!("Unsupported input format. Please use one of the following : svs, ndpi, tiff, vms, vmu, bif, mrxs, scn, svslide, tif, png, jpg, jpeg, tif, tiff, webp, pnm, dds, exr, gif, hdr, ico, tga, bmp, jpg, jpeg, png, ppm, pgm, pnm, pbm, tiff, tif, webp, jp2, j2k, jpf, jpx, jpm, mj2");
            exit(1);
        }
    }
}

fn load_slide()-> openslide_rs::OpenSlide {
    openslide_rs::OpenSlide::new(&ARGS.slide).unwrap()
}

fn load_geometry() -> geojson::FeatureCollection{
    let geometry = File::open(&ARGS.geometry).unwrap();
    let geometry = BufReader::new(geometry);
    let geometry: geojson::FeatureCollection = serde_json::from_reader(geometry).unwrap();
    geometry
}

fn main(){
    // Preping the environment
    ARGS.handle_verbose();
    pretty_env_logger::init();
    ARGS.validate_paths();
    ARGS.validate_gpu();
    ARGS.handle_thread_count();

    // Loading the json file containing the geometry
    info!("Loading the geojson");
    let start = std::time::Instant::now();
    let geometry = load_geometry();
    debug!("Loaded geojson in {:?}", start.elapsed());

    // Loading the slide
    let input_image = load_input_image();
    

    let Args {  output, patch_size, gpus, batch_size, feature_sets, .. } = ARGS.clone();
    let patch_size = patch_size as usize;
    
    let feature_sets = args::FeatureSet::to_fs(&feature_sets);

    let output_df = Arc::new(Mutex::new(None));

    info!("Extracting features");
    let count = geometry.features.len();
    let done = AtomicU32::new(0);
    geometry.features
        .par_chunks(batch_size)
        .map(move |nuclei| {
            match &input_image {
                InputImage::Slide(slide) => utils::load_slide_dataset(nuclei, slide.clone(), patch_size),
                InputImage::Image(image) => utils::load_image_dataset(nuclei, image.clone(), patch_size)
            }
        })
        .map(|x| utils::move_tensors_to_device(x, gpus.clone()))
        .map(|(centroids, polygones, patches, masks)|{
            let features = feature_sets.iter()
                .map(|fs|{
                    debug!("Computing {} features for {} patches (thrd:{})", fs.name(), centroids.len(), rayon::current_thread_index().unwrap_or(0));
                    let start = std::time::Instant::now();
                    let features = fs.compute_features_batched(&centroids, &polygones, &patches, &masks);
                    debug!("Computed {} features for {} patches in {:?} (thrd:{})", fs.name(), centroids.len(), start.elapsed(), rayon::current_thread_index().unwrap_or(0));
                    features
                })
                .collect::<Vec<_>>();

            let centroids = features[0].column("centroid").unwrap().clone();
            features.iter().for_each(|df|assert!(df.column("centroid").unwrap().eq(&centroids)));
            let features = std::iter::once(centroids)
                .chain(
                    features.into_iter()
                        .map(|df|df.drop("centroid").unwrap())
                        .flat_map(|df| df.iter().cloned().collect::<Vec<_>>())
                )
                .collect::<Vec<_>>();

            DataFrame::new(features).unwrap()
        })
        .for_each(|features|{
            let height = features.height();
            let mut output_df = output_df.lock().unwrap();
            if output_df.is_none() {
                *output_df = Some(features);
            } else {
                let output_df = output_df.as_mut().unwrap();
                output_df.vstack_mut(&features).unwrap();
            }
            let done = done.fetch_add(height as u32, std::sync::atomic::Ordering::Relaxed);
            info!("{} / {}", done + height as u32, count);
        });

    let mut output_df = output_df.lock().unwrap().take().unwrap();
    
    let ofile = std::fs::File::create(&output).unwrap();
    let ofile = BufWriter::new(ofile);

    match output.extension().and_then(|ext|ext.to_str()) {
        Some("csv") => {
            let mut writer = polars::io::csv::CsvWriter::new(ofile);
            writer.finish(&mut output_df).unwrap();
        }
        Some("parquet") | Some("pqt") => {
            let writer = polars::io::parquet::ParquetWriter::new(ofile);
            writer.finish(&mut output_df).unwrap();
        }
        Some("json") => {
            let mut writer = polars::io::json::JsonWriter::new(ofile);
            writer.finish(&mut output_df).unwrap();
        }
        Some("ipc") | Some("feather") => {
            let mut writer = polars::io::ipc::IpcWriter::new(ofile);
            writer.finish(&mut output_df).unwrap();
        }
        None => {
            error!("Output file must have an extension");
            exit(1);
        },
        _ => {
            error!("Unsupported output format. Please use one of the following : csv, parquet, json, ipc, feather");
            exit(1);
        }
    }

}
