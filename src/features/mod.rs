mod color;
mod shape;
mod texture;

use polars::prelude::DataFrame;
use tch::Tensor;

/**
 * A feature set is a set of features that can be computed on a batch of patches
 * They are grouped together by type because they often share some computations
 */
pub trait FeatureSet: Send + Sync {
    /**
     * The name of the feature set
     */
    fn name(&self) -> &str;

    /**
     * Compute the features for a batch of patches
     */
    fn compute_features_batched(
        &self,
        centroids: &[[f32; 2]],
        polygons: &[Vec<[f32; 2]>],
        patchs: &Tensor,
        masks: &Tensor,
    ) -> DataFrame;
}

pub use color::ColorFeatureSet;
pub use shape::ShapeFeatureSet;
pub use texture::GLRLMFeatureSet;
pub use texture::GaborFilterFeatureSet;
pub use texture::GlcmFeatureSet;
