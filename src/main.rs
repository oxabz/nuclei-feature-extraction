mod geojson;
mod features;
pub mod consts;
mod utils;

use std::{path::PathBuf, fs::File, io::BufReader, sync::{Arc, RwLock, Mutex, atomic::{AtomicU32, Ordering}}, marker::PhantomData, vec::IntoIter};

use clap::Parser;
use csv::WriterBuilder;
use image::DynamicImage;
use lazy_static::*;
//use itertools::Itertools;
use openslide::OpenSlide;
use rayon::{prelude::*};
use tch::{Tensor, index::*};
use tch_utils::image::ImageTensorExt;

use crate::{consts::PATCH_SIZE, utils::IntoParBridgeBridge, geojson::Feature, features::Features};

#[derive(Debug, Parser)]
struct Args{
    /// Input GeoJSON file
    input_geojson: PathBuf,
    /// Input slide file
    input_slide: PathBuf,
    /// Output file (.csv)
    output: PathBuf
}

impl std::fmt::Display for Args{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Args:\n\tinput_geojson: {:?}\n\tinput_slide: {:?}\n\toutput: {:?}", self.input_geojson, self.input_slide, self.output)
    }
}

lazy_static! {
    static ref SLIDE: Mutex<Option<OpenSlide>> = Mutex::new(None);
}

fn main() {
    let args = Args::parse();
    println!("{}", args);
    
    let file = match File::open(args.input_geojson) {
        Ok(ok) => BufReader::new(ok),
        Err(err) => {
            println!("Couldn't open files : {}", err);
            std::process::exit(1);
        },
    };
    let geojson: geojson::FeatureCollection = match serde_json::from_reader(file) {
        Ok(ok) => {ok},
        Err(err) => {
            println!("Couldn't parse GeoJSON : {}", err);
            std::process::exit(1);
        },
    };

    let slide = match OpenSlide::new(&args.input_slide) {
        Ok(ok) => ok,
        Err(err) => {
            println!("Couldn't open slide : {}", err);
            std::process::exit(1);
        },
    };

    {
        let mut lck = SLIDE.lock().unwrap();
        *lck = Some(slide);
    };

    let done = AtomicU32::new(0);
    let count = geojson.features.len();

    let features = geojson.features.into_par_iter()
        .filter_map(move |feature|{
            let poly = feature.geometry.coordinates[0].clone();
            let mut centroid = (0.0, 0.0);
            for point in &poly{
                centroid.0 += point[0];
                centroid.1 += point[1];
            }
            centroid.0 /= poly.len() as f32;
            centroid.1 /= poly.len() as f32;
            let centered_poly = poly.iter().map(|point|{
                [point[0] - centroid.0, point[1] - centroid.1]
            }).collect::<Vec<_>>();

            let patch = {
                let slide = SLIDE.lock().unwrap();
                let top = centroid.1 - (PATCH_SIZE as f32 / 2.0);
                let left = centroid.0 - (PATCH_SIZE as f32 / 2.0);
                match &*slide{
                    None => return None,
                    Some(slide) =>{
                        slide.read_region(top as usize, left as usize, 0, PATCH_SIZE, PATCH_SIZE)
                    }
                }  
            };
            let patch = match patch {
                Ok(ok) => ok,
                Err(err) => {
                    eprintln!("{err}");
                    return None;
                },
            };

            Some((centroid, centered_poly, patch))
        })
        .map(|(centroid, centered_poly, patch)|{
            let patch = Tensor::from_image(patch.into());
            let patch = patch.to_device(tch::Device::cuda_if_available());
            let patch = patch.i(0..3);

            let features = features::all_features(&centered_poly, &patch);
            let i = done.fetch_add(1, Ordering::Relaxed);
            println!("\x1B[F{} / {}", i, count);

            (centroid, features)
        })
        .collect::<Vec<_>>();

    let mut writer = WriterBuilder::new().from_path(args.output).unwrap();
    Features::write_header_to_csv(&mut writer);
    for (centroid, mut features) in features{
        features.centroid_x = centroid.0;
        features.centroid_y = centroid.1;
        writer.serialize(features).unwrap();
    }
    
}
