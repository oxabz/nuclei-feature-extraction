mod args;
mod geojson;
mod utils;
mod features;
use std::{fs::File, io::{BufReader, BufWriter}, sync::{Arc, Mutex, atomic::{AtomicU32}}, process::exit};
use log::{error, debug};
use args::{ARGS, Args};
use openslide_rs::{traits::Slide, Region, Size};
use polars::prelude::*;
use rayon::prelude::*;
use tch::{Tensor, IndexOp};
use tch_utils::image::ImageTensorExt;


fn load_slide()-> openslide_rs::OpenSlide {
    let slide = openslide_rs::OpenSlide::new(&ARGS.slide).unwrap();
    slide
}

fn load_geometry() -> geojson::FeatureCollection{
    let geometry = File::open(&ARGS.geometry).unwrap();
    let geometry = BufReader::new(geometry);
    let geometry: geojson::FeatureCollection = serde_json::from_reader(geometry).unwrap();
    geometry
}

fn main(){
    // Preping the environment
    pretty_env_logger::init();
    ARGS.handle_verbose();
    ARGS.validate_paths();
    ARGS.validate_gpu();
    ARGS.handle_thread_count();

    // Loading the json file containing the geometry
    let geometry = load_geometry();

    // Loading the slide
    let slide = load_slide();
    let slide = Arc::new(Mutex::new(slide));

    let Args {  output, patch_size, gpus, batch_size, feature_sets, .. } = ARGS.clone();
    let patch_size = patch_size as usize;
    
    let feature_sets = args::FeatureSet::to_fs(&feature_sets);

    let output_df = Arc::new(Mutex::new(None));

    let count = geometry.features.len();
    let done = AtomicU32::new(0);
    geometry.features
        .par_chunks(batch_size)
        .map(|nuclei| utils::load_slides(nuclei, slide.clone(), patch_size))
        .map(|x| utils::move_tensors_to_device(x, gpus.clone()))
        .map(|(centroids, polygones, patches, masks)|{
            let features = feature_sets.iter()
                .map(|fs|{
                    let start = std::time::Instant::now();
                    let features = fs.compute_features_batched(&centroids, &polygones, &patches, &masks);
                    debug!("Computed {} features for {} patches in {:?}", fs.name(), centroids.len(), start.elapsed());
                    features
                })
                .reduce(|a, b| a.join(&b, ["centroid"], ["centroid"], polars::prelude::JoinType::Inner, None).unwrap());
            features.unwrap()
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
            println!("{} / {}", done, count);
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
