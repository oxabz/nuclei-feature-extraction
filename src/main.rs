mod args;
mod geojson;
mod utils;
mod features;
mod consts;
use std::{fs::File, io::{BufReader, BufWriter}, sync::{Arc, Mutex, atomic::{AtomicU32, AtomicUsize}}, process::exit, iter::zip, path::Path};
use features::{ShapeFeatures, color_features};
use log::error;
use args::{ARGS, Args};
use rayon::prelude::*;
use tch::{Kind, Device, Tensor};
use tch_utils::{image::ImageTensorExt, color};
use utils::{PointsExt, preprocess_polygon};


fn load_slide()-> openslide::OpenSlide {
    let slide = openslide::OpenSlide::new(&ARGS.slide).unwrap();
    slide
}

fn load_geometry() -> geojson::FeatureCollection{
    let geometry = File::open(&ARGS.geometry).unwrap();
    let geometry = BufReader::new(geometry);
    let geometry: geojson::FeatureCollection = serde_json::from_reader(geometry).unwrap();
    geometry
}

fn open_output(path: &Path) -> Arc<Mutex<csv::Writer<BufWriter<File>>>> {
    let output = File::create(path).unwrap();
    let output = BufWriter::new(output);
    let mut output = csv::WriterBuilder::default()
        .from_writer(output);
    if let Err(err) = ShapeFeatures::write_header_to_csv(&mut output) {
        error!("Error while writing to csv : {}", err);
        exit(1);
    }
    Arc::new(Mutex::new(output))
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

    match ARGS.feature_set {
        args::FeatureSet::Geometry => geometry_main(geometry),
        args::FeatureSet::Color => color_main(geometry, slide),
        args::FeatureSet::Glcm => todo!(),
    }
}

fn geometry_main(geometry: geojson::FeatureCollection){
    let Args{
        output,
        patch_size,
        ..
    } = ARGS.clone();
    let patch_size = patch_size as usize;

    let output = open_output(&output);

    let count = geometry.features.len();
    let done = AtomicU32::new(0);
    geometry.features.par_iter()
        .map(preprocess_polygon)
        .map(|(centroid, centered_polygone)|{
            let mask = tch_utils::shapes::polygon(patch_size, patch_size, &centered_polygone.to_tchutils_points(), (Kind::Float, Device::Cpu));

            let features = features::shape_features(&centered_polygone, &mask);
            
            (centroid, features)
        })
        .for_each(|(centroid, mut features)|{
            let mut output = output.lock().unwrap();
            features.centroid_x = centroid[0];
            features.centroid_y = centroid[1];
            match output.serialize(features) {
                Ok(_) => {},
                Err(err) => {
                    error!("Error while writing to csv : {}", err);
                },
            };
            let done = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if done % 100 == 0 {
                println!("{} / {}", done, count);
            }
        });
}

fn color_main(geometry: geojson::FeatureCollection, slide: Arc<Mutex<openslide::OpenSlide>>){
    let Args {  output, patch_size, gpus, batch_size, .. } = ARGS.clone();
    let patch_size = patch_size as usize;

    let output = open_output(&output);

    let count = geometry.features.len();
    let done = AtomicU32::new(0);
    let gpu_idx = AtomicUsize::new(0);
    geometry.features
        .par_chunks(batch_size)
        .map(|nuclei|{
            let (centroid_err, patch_mask) : (Vec<_>, Vec<_>) = nuclei.into_iter()
                .map(preprocess_polygon)
                .map(|(centroid, centered_polygone)|{
                    let mask = tch_utils::shapes::polygon(patch_size, patch_size, &centered_polygone.to_tchutils_points(), (Kind::Float, Device::Cpu));
                    let patch = {
                        let slide = slide.lock().unwrap();
                        let left = (centroid[0] - patch_size as f32 / 2.0) as usize;
                        let top = (centroid[1] - patch_size as f32 / 2.0) as usize;
                        slide.read_region(left, top, 0, patch_size, patch_size)
                    };
                    match patch {
                        Ok(ok) => ((centroid, false), (Tensor::from_image(ok.into()), mask)),
                        Err(err) => {
                            error!("Error while reading patch for nuclei {centroid:?} : {}", err);
                            ((centroid, true), (Tensor::ones(&[3, patch_size as i64, patch_size as i64], (Kind::Float, Device::Cpu)), mask))
                        },
                    }
                })
                .unzip();
            let (centroids, err): (Vec<_>, Vec<_>) = centroid_err.into_iter().unzip();
            let (patches, masks):(Vec<_>, Vec<_>) = patch_mask.into_iter().unzip();
            let mut patches = Tensor::stack(&patches, 0);
            let mut masks = Tensor::stack(&masks, 0);
            
            if let Some (gpus) = &gpus {
                let gpus = gpus.clone();
                let gpu_count = gpus.len();
                let gpu_idx = gpu_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                patches = patches.to_device(Device::Cuda(gpus[gpu_idx % gpu_count]));
                masks = masks.to_device(Device::Cuda(gpus[gpu_idx % gpu_count]));
            }
            
            (centroids, err, patches, masks)
        })
        .map(|(centroids, err, patches, masks)|{
            let color_features = color_features(&patches, &masks);
            (centroids, err, color_features)
        })
        .for_each(|(centroids, err, color_features)|{
            let res = zip(centroids, zip(err, color_features));
            let mut output = output.lock().unwrap();
            let done = done.fetch_add(res.len() as u32, std::sync::atomic::Ordering::Relaxed);
            for (centroid, (err, mut features)) in res {
                if err {
                    features.set_all_nan();
                }
                features.centroid_x = centroid[0];
                features.centroid_y = centroid[1];
                match output.serialize(features) {
                    Ok(_) => {},
                    Err(err) => {
                        error!("Error while writing to csv : {}", err);
                    },
                };
            }
            println!("{} / {}", done, count);
        });
}

fn glcm_main(geometry: geojson::FeatureCollection, slide: Arc<Mutex<openslide::OpenSlide>>){
    
}