mod args;
mod geojson;
mod utils;
mod features;
mod consts;
use std::{fs::File, io::{BufReader, BufWriter}, sync::{Arc, Mutex, atomic::AtomicU32}, process::exit};
use features::ShapeFeatures;
use log::error;
use object_pool::Pool;
use args::{ARGS, Args};
use rayon::prelude::*;
use tch::{Kind, Device};
use utils::{PointsExt};

fn load_slide()-> Pool<openslide::OpenSlide> {
    let pool = Pool::new(ARGS.openslide_instance_count, || {
        let slide = openslide::OpenSlide::new(&ARGS.slide).unwrap();
        slide
    });
    pool
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
    ARGS.validate_paths();
    ARGS.validate_gpu();
    ARGS.handle_thread_count();

    // Loading the json file containing the geometry
    let geometry = load_geometry();

    // Loading the slide
    let slide = load_slide();
    let slide = Arc::new(slide);

    match ARGS.feature_set {
        args::FeatureSet::Geometry => geometry_main(geometry),
        args::FeatureSet::Color => todo!(),
        args::FeatureSet::Texture => todo!(),
    }
}

fn geometry_main(geometry: geojson::FeatureCollection){
    let Args{
        output,
        patch_size,
        ..
    } = ARGS.clone();
    let patch_size = patch_size as usize;

    let output = File::create(output).unwrap();
    let output = BufWriter::new(output);
    let mut output = csv::WriterBuilder::default()
        .from_writer(output);
    if let Err(err) = ShapeFeatures::write_header_to_csv(&mut output) {
        error!("Error while writing to csv : {}", err);
        exit(1);
    }
    let output = Arc::new(Mutex::new(output));

    let count = geometry.features.len();
    let done = AtomicU32::new(0);
    geometry.features.into_par_iter()
        .map(|feature|{
            let polynome = &feature.geometry.coordinates[0];
            let centroid = polynome.iter().fold([0.0,0.0], |mut acc, point| {
                acc[0] += point[0];
                acc[1] += point[1];
                acc
            });
            let centroid = [centroid[0] / polynome.len() as f32, centroid[1] / polynome.len() as f32];
            let centered_polygone = polynome.iter().map(|point|{
                let x = point[0] - centroid[0];
                let y = point[1] - centroid[1];
                [x,y]
            }).collect::<Vec<_>>();
            (centroid, centered_polygone)
        })
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
