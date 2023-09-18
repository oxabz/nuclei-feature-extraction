use std::sync::{Arc, Mutex};
use tch::Tensor;

use crate::geojson::Feature;
use crate::utils::{self, Batch};

#[derive(Clone)]
pub enum InputImage {
    Slide(Arc<Mutex<openslide_rs::OpenSlide>>),
    Image(Arc<Mutex<Tensor>>),
}

pub(crate) type BatchLoader = Box<dyn for<'a> Fn(&'a [Feature]) -> Batch + Send + Sync>;
impl InputImage {
    pub(crate) fn patch_loader(&self, patch_size: usize) -> BatchLoader {
        match self {
            InputImage::Slide(slide) => {
                let slide = slide.clone();
                Box::new(move |features: &[Feature]| {
                    utils::load_slide_dataset(features, slide.clone(), patch_size)
                }) as BatchLoader
            }
            InputImage::Image(image) => {
                let image = image.clone();
                Box::new(move |features: &[Feature]| {
                    utils::load_image_dataset(features, image.clone(), patch_size)
                }) as BatchLoader
            }
        }
    }
}
