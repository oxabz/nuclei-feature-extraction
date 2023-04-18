mod geojson;
mod features;
pub mod consts;
mod utils;

use std::{path::PathBuf, fs::File, io::BufReader};

use clap::Parser;
use csv::WriterBuilder;
use openslide::OpenSlide;
use rayon::prelude::*;
use tch::Tensor;

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

    let res = geojson.features.par_iter().map(|feature|{
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

        let features = features::shape_features(&centered_poly, &Tensor::of_slice(&[0.0]));
        
        (centroid, features)
    }).collect::<Vec<_>>();

    let mut writter = WriterBuilder::new()
        .has_headers(true)
        .from_path(args.output)
        .unwrap();
    writter.write_record(&["centroid_x", "centroid_y", "area", "major_axis", "minor_axis"]).expect("Couldn't write headers");

    for (centroid, features) in res{
        writter.write_record(&[
            centroid.0.to_string(),
            centroid.1.to_string(),
            features.area.to_string(),
            features.major_axis.to_string(),
            features.minor_axis.to_string(),
        ]).expect("Couldn't write record");
    }

    // (0..100)
    //     .into_par_iter()
    //     .map(|i| {
    //         let poly = geojson.features[i].geometry.coordinates[0].clone();
    //         let mut centroid = (0.0, 0.0);
    //         for point in &poly{
    //             centroid.0 += point[0];
    //             centroid.1 += point[1];
    //         }
    //         centroid.0 /= poly.len() as f32;
    //         centroid.1 /= poly.len() as f32;

    //         let centered_poly = poly.iter().map(|point|{
    //             [point[0] - centroid.0, point[1] - centroid.1]
    //         }).collect::<Vec<_>>();

    //         let mask = mask::poly2mask((PATCH_SIZE, PATCH_SIZE), &centered_poly);
    //         let mask = Tensor::of_slice(&mask).view((1, 1, PATCH_SIZE as i64, PATCH_SIZE as i64)).to_kind(Kind::Float);

    //         let conv_hull = mask::poly2mask_convex((PATCH_SIZE, PATCH_SIZE), &centered_poly);
    //         let conv_hull = Tensor::of_slice(&conv_hull).view((1, 1, PATCH_SIZE as i64, PATCH_SIZE as i64)).to_kind(Kind::Float);

    //         let c = Tensor::cat(&[mask, conv_hull], 3);

    //         tch::vision::image::save(&c, format!("masks/mask_{}.png", i)).expect("Couldn't save mask");


    //     })
    //     .collect::<Vec<_>>();

}
