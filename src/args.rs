use clap::Parser;
use log::error;
use std::{path::PathBuf, process::exit};

use crate::features;

#[derive(Clone, Debug)]
pub enum FeatureSet {
    Geometry,
    Color,
    Glcm,
    Glrlm,
    Gabor,
    Texture,
    All,
}

impl std::str::FromStr for FeatureSet {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "geometry" => Ok(FeatureSet::Geometry),
            "color" => Ok(FeatureSet::Color),
            "glcm" => Ok(FeatureSet::Glcm),
            "glrlm" => Ok(FeatureSet::Glrlm),
            "gabor" => Ok(FeatureSet::Gabor),
            "texture" => Ok(FeatureSet::Texture),
            "all" => Ok(FeatureSet::All),
            _ => Err(format!("{} is not a valid feature set", s)),
        }
    }
}

impl FeatureSet {
    pub fn flat(s: &Vec<Self>) -> Vec<Self> {
        s.iter()
            .flat_map(|fs| match fs {
                FeatureSet::All => vec![
                    FeatureSet::Geometry,
                    FeatureSet::Color,
                    FeatureSet::Glcm,
                    FeatureSet::Glrlm,
                    FeatureSet::Gabor,
                ],
                FeatureSet::Texture => vec![FeatureSet::Glcm, FeatureSet::Glrlm, FeatureSet::Gabor],
                fs => vec![fs.clone()],
            })
            .collect()
    }

    pub fn to_fs(s: &Vec<Self>) -> Vec<Box<dyn features::FeatureSet>> {
        let fs = Self::flat(s);
        fs.iter()
            .map(|fs| match fs {
                FeatureSet::Geometry => {
                    Box::new(features::ShapeFeatureSet) as Box<dyn features::FeatureSet>
                }
                FeatureSet::Color => {
                    Box::new(features::ColorFeatureSet) as Box<dyn features::FeatureSet>
                }
                FeatureSet::Glcm => {
                    Box::new(features::GlcmFeatureSet) as Box<dyn features::FeatureSet>
                }
                FeatureSet::Glrlm => {
                    Box::new(features::GLRLMFeatureSet) as Box<dyn features::FeatureSet>
                }
                FeatureSet::Gabor => {
                    Box::new(features::GaborFilterFeatureSet) as Box<dyn features::FeatureSet>
                }
                FeatureSet::All | FeatureSet::Texture => unreachable!(),
            })
            .collect()
    }
}

#[derive(Debug, Parser, Clone)]
pub struct Args {
    /// Input geometry file (.geojson)
    pub geometry: PathBuf,
    /// Input slide file (.svs)
    pub slide: PathBuf,
    /// Output file (polars compatible formats) if the file exist it will be completed unless --overwrite is specified
    /// Supported formats: see https://pola-rs.github.io/polars/py-polars/html/reference/io.html
    pub output: PathBuf,
    /// Feature sets to extract
    pub feature_sets: Vec<FeatureSet>,
    /// Overwrite :
    /// if specified, will overwrite the output file if it already exists
    #[clap(short, long)]
    pub overwrite: bool,
    /// Patch size :
    /// the size of the patch to extract from the slide (in pixels)
    #[clap(short, long, default_value = "64")]
    pub patch_size: u32,
    /// Thread count :
    /// the number of threads used by rayon
    /// if not specified, rayon will use the number of cores available on the machine
    #[clap(short, long)]
    pub thread_count: Option<usize>,
    /// gpus :
    /// if specified, will use the specified gpus to extract the patches
    /// if not specified, will use the cpu
    #[clap(short, long)]
    pub gpus: Option<Vec<usize>>,
    /// batch size :
    /// the number of patches to extract at once
    #[clap(short, long, default_value = "1000")]
    pub batch_size: usize,
    /// verbose :
    /// if specified, will print more information
    #[clap(short, long)]
    pub verbose: bool,
}

impl Args {
    pub fn handle_verbose(&self) {
        if !self.verbose {
            return;
        }
        println!("Called Args :");
        println!("{:#?}", self);
        log::set_max_level(log::LevelFilter::Trace);
    }

    pub fn handle_thread_count(&self) {
        if let Some(thread_count) = self.thread_count {
            rayon::ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build_global()
                .unwrap();
        }
    }

    pub fn validate_paths(&self) {
        if !self.geometry.exists() {
            error!("Geometry file does not exist : {:?}", self.geometry);
            exit(1);
        }
        if !self.slide.exists() {
            error!("Slide file does not exist : {:?}", self.slide);
            exit(1);
        }

        if self.output.exists() {
            if !self.overwrite {
                error!(
                    "Output file already exists : {:?}\nUse --overwrite to overwrite it",
                    self.output
                );
                exit(1);
            }
        }

        match self.output.extension().and_then(|s| s.to_str()) {
            Some("csv") | Some("parquet") | Some("pqt") |
            Some("json") | Some("ipc") | Some("feather") => {}
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

    pub fn validate_gpu(&self) {
        if let Some(gpus) = &self.gpus {
            if !tch::Cuda::is_available() {
                error!("No GPU available\nCheck that CUDA is installed and that your GPU is compatible with CUDA\nCheck that you specified the right version of libtorch in LIBTORCH and LD_LIBRARY_PATH");
                exit(1);
            }
            let device_count = tch::Cuda::device_count();
            for gpu in gpus {
                if *gpu >= device_count as usize {
                    error!("GPU {} does not exist", gpu);
                    exit(1);
                }
            }
        }
    }
}

lazy_static::lazy_static! {
    pub static ref ARGS: Args = Args::parse();
}
