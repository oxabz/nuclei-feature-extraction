use std::{path::PathBuf, process::exit};
use clap::Parser;
use log::error;

#[derive(Clone, Debug)]
pub enum FeatureSet{
    Geometry, 
    Color,
    Glcm,
    Glrlm,
}

impl std::str::FromStr for FeatureSet{
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "geometry" => Ok(FeatureSet::Geometry),
            "color" => Ok(FeatureSet::Color),
            "glcm" => Ok(FeatureSet::Glcm),
            "glrlm" => Ok(FeatureSet::Glrlm),
            _ => Err(format!("{} is not a valid feature set", s)),
        }
    }
}

#[derive(Debug, Parser, Clone)]
pub struct Args{
    /// Feature set to extract
    pub feature_set: FeatureSet,
    /// Input geometry file (.geojson)
    pub geometry: PathBuf,
    /// Input slide file (.svs)
    pub slide: PathBuf,
    /// Output file (polars compatible formats) if the file exist it will be completed unless --overwrite is specified
    /// Supported formats: see https://pola-rs.github.io/polars/py-polars/html/reference/io.html
    pub output: PathBuf,
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

impl Args{
    pub fn handle_verbose(&self){
        if !self.verbose{return}
        println!("Called Args :");
        println!("{:#?}", self.feature_set);
        log::set_max_level(log::LevelFilter::Debug);
    }

    pub fn handle_thread_count(&self){
        if let Some(thread_count) = self.thread_count {
            rayon::ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build_global()
                .unwrap();
        }
    }

    pub fn validate_paths(&self){
        if !self.geometry.exists(){
            error!("Geometry file does not exist : {:?}", self.geometry);
            exit(1);
        }
        if !self.slide.exists(){
            error!("Slide file does not exist : {:?}", self.slide);
            exit(1);
        }

        if self.output.exists(){
            if !self.overwrite{
                error!("Output file already exists : {:?}\nUse --overwrite to overwrite it", self.output);
                exit(1);
            }
        }
    }

    pub fn validate_gpu(&self){
        if let Some(gpus) = &self.gpus{
            if !tch::Cuda::is_available() {
                error!("No GPU available\nCheck that CUDA is installed and that your GPU is compatible with CUDA\nCheck that you specified the right version of libtorch in LIBTORCH and LD_LIBRARY_PATH");
                exit(1);
            }
            let device_count = tch::Cuda::device_count();
            for gpu in gpus{
                if *gpu >= device_count as usize{
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