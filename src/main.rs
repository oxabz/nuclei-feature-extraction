mod args;
mod features;
mod geojson;
mod input;
mod utils;
use args::{Args, ARGS};
use input::InputImage;
use log::{debug, error, info};
use polars::prelude::*;
use rayon::prelude::*;
use std::{
    fs::File,
    io::{BufReader, BufWriter},
    process::exit,
    sync::{atomic::AtomicU32, Arc, Mutex},
};

use crate::utils::Batch;

fn load_input_image() -> InputImage {
    match ARGS.slide.extension().and_then(|ext| ext.to_str()) {
        Some("svs") => InputImage::Slide(Arc::new(Mutex::new(
            openslide_rs::OpenSlide::new(&ARGS.slide).unwrap(),
        ))),
        Some("png") | Some("jpg") | Some("jpeg") => InputImage::Image(Arc::new(Mutex::new(
            tch::vision::image::load(&ARGS.slide).unwrap(),
        ))),
        _ => {
            error!("Unsupported input format. Please use one of the following : svs, png, jpg, jpeg");
            exit(1);
        }
    }
}

fn load_geometry() -> geojson::FeatureCollection {
    let geometry = File::open(&ARGS.geometry).unwrap();
    let geometry = BufReader::new(geometry);
    let geometry: geojson::FeatureCollection = serde_json::from_reader(geometry).unwrap();
    geometry
}

/**
Returns a closure that takes a batch of patches and returns a dataframe containing the features
 */
fn extract_features<'a>(
    feature_sets: &'a [Box<dyn features::FeatureSet>],
) -> impl Fn(Batch) -> DataFrame + Send + Sync + 'a {
    move |(centroids, polygones, patches, masks)| {
        // Computing the features for each batch of patches
        let features = feature_sets
            .iter()
            .map(|fs| {
                debug!(
                    "Computing {} features for {} patches (thrd:{})",
                    fs.name(),
                    centroids.len(),
                    rayon::current_thread_index().unwrap_or(0)
                );
                let start = std::time::Instant::now();
                let features =
                    fs.compute_features_batched(&centroids, &polygones, &patches, &masks);
                debug!(
                    "Computed {} features for {} patches in {:?} (thrd:{})",
                    fs.name(),
                    centroids.len(),
                    start.elapsed(),
                    rayon::current_thread_index().unwrap_or(0)
                );
                features
            })
            .collect::<Vec<_>>();

        // Concatenating the features
        let centroids = features[0].column("centroid").unwrap().clone();
        features
            .iter()
            .for_each(|df| assert!(df.column("centroid").unwrap().eq(&centroids)));
        let features = std::iter::once(centroids)
            .chain(
                features
                    .into_iter()
                    .map(|df| df.drop("centroid").unwrap())
                    .flat_map(|df| df.iter().cloned().collect::<Vec<_>>()),
            )
            .collect::<Vec<_>>();

        DataFrame::new(features).unwrap()
    }
}

/**
Returns a closure that append the extracted features of a batch to the output dataframe
 */
fn append_features(
    output_df: Arc<Mutex<Option<DataFrame>>>,
) -> impl Fn(DataFrame) -> () + Send + Sync {
    move |df| {
        let mut output_df = output_df.lock().unwrap();
        if output_df.is_none() {
            *output_df = Some(df);
        } else {
            let output_df = output_df.as_mut().unwrap();
            output_df.vstack_mut(&df).unwrap();
        }
    }
}

fn main() {
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

    // Loading the input image
    let input_image = load_input_image();

    // Unpacking the arguments
    let Args {
        output,
        patch_size,
        gpus,
        batch_size,
        feature_sets,
        ..
    } = ARGS.clone();
    let patch_size = patch_size as usize;
    let feature_sets = args::FeatureSet::to_fs(&feature_sets);

    // Initializing the output dataframe
    let output_df = Arc::new(Mutex::new(None));

    // Extracting the features
    info!("Extracting features");
    let count = geometry.features.len();
    let done = AtomicU32::new(0);
    geometry
        .features
        .par_chunks(batch_size) // Splitting the geometry into batches
        .map(input_image.patch_loader(patch_size)) // Loading the patches for each batch we use a closure to avoid having to use a conditional
        .map(|x: Batch| utils::move_tensors_to_device(x, gpus.clone())) // Moving the tensors to the GPU
        .map(extract_features(&feature_sets)) // Extracting the features
        .map(|df| {
            // Logging the progress
            let done = done.fetch_add(df.height() as u32, std::sync::atomic::Ordering::SeqCst);
            info!("Extracted features for {}/{} patches", done, count);
            df
        })
        .for_each(append_features(output_df.clone())); // Appending the features to the output dataframe

    // Saving the output dataframe
    let mut output_df = output_df.lock().unwrap().take().unwrap();
    let ofile = std::fs::File::create(&output).unwrap();
    let ofile = BufWriter::new(ofile);
    match output.extension().and_then(|ext| ext.to_str()) {
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
        }
        _ => {
            error!("Unsupported output format. Please use one of the following : csv, parquet, json, ipc, feather");
            exit(1);
        }
    }
}
