mod color;
mod shape;
mod texture;

use polars::prelude::DataFrame;
use tch::Tensor;

pub trait FeatureSet : Send + Sync {
    fn name(&self)->&str;

    fn compute_features_batched(&self, centroids: &[[f32; 2]], polygons: &[Vec<[f32; 2]>], patchs: &Tensor, masks: &Tensor) -> DataFrame;
}

pub use color::ColorFeatureSet;
pub use shape::ShapeFeatureSet;
pub use texture::GlcmFeatureSet;
pub use texture::GLRLMFeatureSet;
pub use texture::GaborFilterFeatureSet;
