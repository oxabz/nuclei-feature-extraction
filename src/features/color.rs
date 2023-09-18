use polars::prelude::*;
use tch::Tensor;

use crate::utils::centroids_to_key_strings;

use super::FeatureSet;

pub struct ColorFeatureSet;

impl FeatureSet for ColorFeatureSet {
    fn compute_features_batched(
        &self,
        centroids: &[[f32; 2]],
        polygons: &[Vec<[f32; 2]>],
        patchs: &Tensor,
        masks: &Tensor,
    ) -> polars::prelude::DataFrame {
        assert!(
            patchs.size().len() == 4,
            "The patchs tensor must be 4 dimensional"
        );
        assert!(
            masks.size().len() == 4,
            "The masks tensor must be 4 dimensional"
        );
        assert!(
            patchs.size()[1] == 3,
            "The patchs tensor must have 3 channels"
        );
        assert!(masks.size()[1] == 1, "The masks tensor must have 1 channel");
        assert!(
            patchs.size()[0] == masks.size()[0],
            "The number of patchs and masks must be the same"
        );
        assert!(
            patchs.size()[0] as usize == centroids.len(),
            "The number of patchs and centroids must be the same"
        );
        assert!(
            patchs.size()[0] as usize == polygons.len(),
            "The number of patchs and polygons must be the same"
        );

        let _ = tch::no_grad_guard();
        let hsv = tch_utils::color::hsv_from_rgb(patchs);
        let hed = tch_utils::color::hed_from_rgb(patchs);
        let (mean_rgb, std_rgb) = mean_std(patchs, masks);
        let (mean_hed, std_hed) = mean_std(&hed, masks);

        let mut h = hsv.select(-3, 0);
        let mean_h = circular_mean(&h, masks);
        h -= &mean_h.view([-1, 1, 1]);
        let (mean_hsv, std_hsv) = mean_std(&hsv, masks);

        let mean_r = Vec::<f32>::from(mean_rgb.select(-1, 0));
        let mean_g = Vec::<f32>::from(mean_rgb.select(-1, 1));
        let mean_b = Vec::<f32>::from(mean_rgb.select(-1, 2));

        let std_r = Vec::<f32>::from(std_rgb.select(-1, 0));
        let std_g = Vec::<f32>::from(std_rgb.select(-1, 1));
        let std_b = Vec::<f32>::from(std_rgb.select(-1, 2));

        let mean_h = Vec::<f32>::from(mean_h);
        let mean_s = Vec::<f32>::from(mean_hsv.select(-1, 1));
        let mean_v = Vec::<f32>::from(mean_hsv.select(-1, 2));

        let std_h = Vec::<f32>::from(std_hsv.select(-1, 0));
        let std_s = Vec::<f32>::from(std_hsv.select(-1, 1));
        let std_v = Vec::<f32>::from(std_hsv.select(-1, 2));

        let mean_haematoxylin = Vec::<f32>::from(mean_hed.select(-1, 0));
        let mean_eosin = Vec::<f32>::from(mean_hed.select(-1, 1));
        let mean_dab = Vec::<f32>::from(mean_hed.select(-1, 2));

        let std_haematoxylin = Vec::<f32>::from(std_hed.select(-1, 0));
        let std_eosin = Vec::<f32>::from(std_hed.select(-1, 1));
        let std_dab = Vec::<f32>::from(std_hed.select(-1, 2));

        let centroids = centroids_to_key_strings(centroids);
        df!(
            "centroid" => centroids,
            "mean_r" => mean_r,
            "mean_g" => mean_g,
            "mean_b" => mean_b,
            "std_r" => std_r,
            "std_g" => std_g,
            "std_b" => std_b,
            "mean_h" => mean_h,
            "mean_s" => mean_s,
            "mean_v" => mean_v,
            "std_h" => std_h,
            "std_s" => std_s,
            "std_v" => std_v,
            "mean_haematoxylin" => mean_haematoxylin,
            "mean_eosin" => mean_eosin,
            "mean_dab" => mean_dab,
            "std_haematoxylin" => std_haematoxylin,
            "std_eosin" => std_eosin,
            "std_dab" => std_dab,
        )
        .expect("Could not create the dataframe")
    }

    fn name(&self) -> &str {
        "color"
    }
}

/**
Compute the mean and standard deviation of the color channels of the image.
# Arguments
- `img` - [N, 3, H, W] tensor
- `mask` - [N, 1, H, W] tensor
# Returns
- `mean` - [N, 3] tensor
 */
pub fn mean_std(img: &Tensor, mask: &Tensor) -> (Tensor, Tensor) {
    let masked /* [N, 3, H, W] */ = img * mask;
    let mask_area /* [N, 3, 1, 1] */ = mask.sum_dim_intlist(Some(&[-1i64, -2][..]), true, tch::Kind::Float);
    let mut mean /* [N, 3, 1, 1] */ = masked.sum_dim_intlist( Some(&[-1i64, -2][..]), true, tch::Kind::Float);
    mean /= &mask_area;

    let std /* [N, 3, H, W] */ = img - &mean;
    let std = std.square() * mask;

    let mut std = std.sum_dim_intlist(Some(&[-1i64, -2][..]), true, tch::Kind::Float);
    std /= &mask_area;
    std = std.sqrt();

    let _ = std.squeeze_(); // [N, 3, 1, 1] -> [N, 3]
    let _ = mean.squeeze_(); // [N, 3, 1, 1] -> [N, 3]

    (mean, std)
}

/**
Compute the mean deviation of the image.
# Arguments
- `img` - [N, 1, H, W] tensor
- `mask` - [N, 1, H, W] tensor
# Returns
- `mean` - [N] tensor
 */
pub fn circular_mean(image: &Tensor, mask: &Tensor) -> Tensor {
    let mask_area = mask.sum_dim_intlist(Some(&[-1, -2, -3][..]), false, tch::Kind::Float);

    let img = image.deg2rad();
    let cos = img.cos() * mask;
    let sin = img.sin() * mask;

    let cos = cos.sum_dim_intlist(Some(&[-1, -2, -3][..]), false, tch::Kind::Float) / &mask_area;
    let sin = sin.sum_dim_intlist(Some(&[-1, -2, -3][..]), false, tch::Kind::Float) / &mask_area;

    (sin.atan2(&cos).rad2deg_() + 360.0).fmod_(360.0)
}
