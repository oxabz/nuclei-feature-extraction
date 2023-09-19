use polars::{
    prelude::{DataFrame, NamedFrom},
    series::Series,
};
use tch::{index::*, Kind, Tensor};
use tch_utils::{glcm::{glcm, features::*}, glrlm::{glrlm, features::{GlrlmFeatures, glrlm_features}}, tensor_ext::TensorExt};

use crate::utils::centroids_to_key_strings;

use super::FeatureSet;

const GLCM_LEVELS: [u8; 4] = [32, 64, 128, 254];
const OFFSETS: [(i64, i64); 4] = [(0, 1), (1, 1), (1, 0), (1, -1)]; // 0, 45, 90, 135

pub struct GlcmFeatureSet;

impl FeatureSet for GlcmFeatureSet {
    fn compute_features_batched(
        &self,
        centroids: &[[f32; 2]],
        _polygons: &[Vec<[f32; 2]>],
        patchs: &Tensor,
        masks: &Tensor,
    ) -> polars::prelude::DataFrame {
        let _ = tch::no_grad_guard();

        let mut features = Vec::with_capacity(14 * OFFSETS.len() * GLCM_LEVELS.len());

        let gray_scale_patch = patchs.mean_dim(Some(&([-3][..])), true, Kind::Float); // [N, 1, H, W]

        for gray_level_count in GLCM_LEVELS {
            for offset in OFFSETS {
                let glcm = glcm(&gray_scale_patch, offset, gray_level_count, Some(masks), true);

                let GlcmFeatures { correlation, contrast, dissimilarity, entropy, angular_second_moment, sum_average, sum_variance, sum_entropy, sum_of_squares, inverse_difference_moment, difference_average, difference_variance, information_measure_of_correlation_1, information_measure_of_correlation_2 } = glcm_features(&glcm);
                let correlation = Vec::<f32>::from(correlation);
                let contrast = Vec::<f32>::from(contrast);
                let dissimilarity = Vec::<f32>::from(dissimilarity);
                let entropy = Vec::<f32>::from(entropy);
                let angular_second_moment = Vec::<f32>::from(angular_second_moment);
                let sum_average = Vec::<f32>::from(sum_average);
                let sum_variance = Vec::<f32>::from(sum_variance);
                let sum_entropy = Vec::<f32>::from(sum_entropy);
                let sum_of_squares = Vec::<f32>::from(sum_of_squares);
                let inverse_difference_moment = Vec::<f32>::from(inverse_difference_moment);
                let difference_average = Vec::<f32>::from(difference_average);
                let difference_variance = Vec::<f32>::from(difference_variance);
                let information_measure_correlation1 =
                    Vec::<f32>::from(information_measure_of_correlation_1);
                let information_measure_correlation2 =
                    Vec::<f32>::from(information_measure_of_correlation_2);

                features.push((
                    format!("correlation_{}_{}_{gray_level_count}", offset.0, offset.1),
                    correlation,
                ));
                features.push((
                    format!("contrast_{}_{}_{gray_level_count}", offset.0, offset.1),
                    contrast,
                ));
                features.push((
                    format!("dissimilarity_{}_{}_{gray_level_count}", offset.0, offset.1),
                    dissimilarity,
                ));
                features.push((
                    format!("entropy_{}_{}_{gray_level_count}", offset.0, offset.1),
                    entropy,
                ));
                features.push((
                    format!("angular_second_moment_{}_{}_{gray_level_count}", offset.0, offset.1),
                    angular_second_moment,
                ));
                features.push((
                    format!("sum_average_{}_{}_{gray_level_count}", offset.0, offset.1),
                    sum_average,
                ));
                features.push((
                    format!("sum_variance_{}_{}_{gray_level_count}", offset.0, offset.1),
                    sum_variance,
                ));
                features.push((
                    format!("sum_entropy_{}_{}_{gray_level_count}", offset.0, offset.1),
                    sum_entropy,
                ));
                features.push((
                    format!("sum_of_squares_{}_{}_{gray_level_count}", offset.0, offset.1),
                    sum_of_squares,
                ));
                features.push((
                    format!(
                        "inverse_difference_moment_{}_{}_{gray_level_count}",
                        offset.0, offset.1
                    ),
                    inverse_difference_moment,
                ));
                features.push((
                    format!("difference_average_{}_{}_{gray_level_count}", offset.0, offset.1),
                    difference_average,
                ));
                features.push((
                    format!("difference_variance_{}_{}_{gray_level_count}", offset.0, offset.1),
                    difference_variance,
                ));
                features.push((
                    format!(
                        "information_measure_correlation1_{}_{}_{gray_level_count}",
                        offset.0, offset.1
                    ),
                    information_measure_correlation1,
                ));
                features.push((
                    format!(
                        "information_measure_correlation2_{}_{}_{gray_level_count}",
                        offset.0, offset.1
                    ),
                    information_measure_correlation2,
                ));

            }
        }

        let features =
            std::iter::once(Series::new("centroid", centroids_to_key_strings(centroids)))
                .chain(features.into_iter().map(|(k, v)| Series::new(&k, v)))
                .collect::<Vec<_>>();

        DataFrame::new(features).unwrap()
    }

    fn name(&self) -> &str {
        "GLCM"
    }
}

const GLRLM_LEVELS: u8 = 24;
const GLRLM_MAX_LENGTH: i64 = 16;
const DIRECTIONS: [(i64, i64); 4] = [(1, 0), (1, 1), (0, 1), (-1, 1)];

pub struct GLRLMFeatureSet;

impl FeatureSet for GLRLMFeatureSet {
    fn compute_features_batched(
        &self,
        centroids: &[[f32; 2]],
        _polygons: &[Vec<[f32; 2]>],
        patchs: &Tensor,
        masks: &Tensor,
    ) -> DataFrame {
        let _ = tch::no_grad_guard();
        let gs = patchs.mean_dim(Some(&[-3][..]), true, Kind::Float);

        let features = DIRECTIONS.iter().flat_map(|direction| {
            // Gray Level Run-Length Matrix [N, GLRLM_LEVELS, GLRLM_MAX_LENGTH]
            let glrlm = glrlm(&gs, GLRLM_LEVELS, GLRLM_MAX_LENGTH, *direction, Some(masks))
                .to_kind(Kind::Float);

            let pixel_count = masks.sum_dims([-3, -2, -1]);
            let GlrlmFeatures { run_percentage, run_length_mean, run_length_variance, gray_level_non_uniformity, run_length_non_uniformity, short_run_emphasis, long_run_emphasis, low_gray_level_run_emphasis, high_gray_level_run_emphasis, short_run_low_gray_level_emphasis, short_run_high_gray_level_emphasis, long_run_low_gray_level_emphasis, long_run_high_gray_level_emphasis, short_run_mid_gray_level_emphasis, long_run_mid_gray_level_emphasis, short_run_extreme_gray_level_emphasis, long_run_extreme_gray_level_emphasis } = glrlm_features(&glrlm, Some(&pixel_count));

            vec![
                Series::new(
                    &format!("short_run_emphasis_{}_{}", direction.0, direction.1),
                    Vec::<f32>::from(short_run_emphasis),
                ),
                Series::new(
                    &format!("long_run_emphasis_{}_{}", direction.0, direction.1),
                    Vec::<f32>::from(long_run_emphasis),
                ),
                Series::new(
                    &format!("gray_level_nonuniformity_{}_{}", direction.0, direction.1),
                    Vec::<f32>::from(gray_level_non_uniformity),
                ),
                Series::new(
                    &format!("run_length_nonuniformity_{}_{}", direction.0, direction.1),
                    Vec::<f32>::from(run_length_non_uniformity),
                ),
                Series::new(
                    &format!(
                        "low_gray_level_run_emphasis_{}_{}",
                        direction.0, direction.1
                    ),
                    Vec::<f32>::from(low_gray_level_run_emphasis),
                ),
                Series::new(
                    &format!(
                        "high_gray_level_run_emphasis_{}_{}",
                        direction.0, direction.1
                    ),
                    Vec::<f32>::from(high_gray_level_run_emphasis),
                ),
                Series::new(
                    &format!(
                        "short_run_low_gray_level_emphasis_{}_{}",
                        direction.0, direction.1
                    ),
                    Vec::<f32>::from(short_run_low_gray_level_emphasis),
                ),
                Series::new(
                    &format!(
                        "short_run_high_gray_level_emphasis_{}_{}",
                        direction.0, direction.1
                    ),
                    Vec::<f32>::from(short_run_high_gray_level_emphasis),
                ),
                Series::new(
                    &format!(
                        "long_run_low_gray_level_emphasis_{}_{}",
                        direction.0, direction.1
                    ),
                    Vec::<f32>::from(long_run_low_gray_level_emphasis),
                ),
                Series::new(
                    &format!(
                        "long_run_high_gray_level_emphasis_{}_{}",
                        direction.0, direction.1
                    ),
                    Vec::<f32>::from(long_run_high_gray_level_emphasis),
                ),
                Series::new(
                    &format!(
                        "short_run_mid_gray_level_emphasis_{}_{}",
                        direction.0, direction.1
                    ),
                    Vec::<f32>::from(short_run_mid_gray_level_emphasis),
                ),
                Series::new(
                    &format!(
                        "long_run_mid_gray_level_emphasis_{}_{}",
                        direction.0, direction.1
                    ),
                    Vec::<f32>::from(long_run_mid_gray_level_emphasis),
                ),
                Series::new(
                    &format!(
                        "short_run_extreme_gray_level_emphasis_{}_{}",
                        direction.0, direction.1
                    ),
                    Vec::<f32>::from(short_run_extreme_gray_level_emphasis),
                ),
                Series::new(
                    &format!(
                        "long_run_extreme_gray_level_emphasis_{}_{}",
                        direction.0, direction.1
                    ),
                    Vec::<f32>::from(long_run_extreme_gray_level_emphasis),
                ),
                Series::new(
                    &format!("run_percentage_{}_{}", direction.0, direction.1),
                    Vec::<f32>::from(run_percentage),
                ),
                Series::new(
                    &format!("run_length_mean_{}_{}", direction.0, direction.1),
                    Vec::<f32>::from(run_length_mean),
                ),
                Series::new(
                    &format!("run_length_variance_{}_{}", direction.0, direction.1),
                    Vec::<f32>::from(run_length_variance),
                ),
            ]
        });

        let features =
            std::iter::once(Series::new("centroid", centroids_to_key_strings(centroids)))
                .chain(features)
                .collect::<Vec<_>>();

        DataFrame::new(features).unwrap()
    }

    fn name(&self) -> &str {
        "GLRLM"
    }
}

pub struct GaborFilterFeatureSet;

const ANGLES_COUNT: usize = 8;
const FREQUENCIES: [f64; 6] = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0];

impl FeatureSet for GaborFilterFeatureSet {
    fn compute_features_batched(
        &self,
        centroids: &[[f32; 2]],
        _polygons: &[Vec<[f32; 2]>],
        patchs: &Tensor,
        masks: &Tensor,
    ) -> DataFrame {
        let _ = tch::no_grad_guard();

        let gs = patchs.mean_dim(Some(&[-3][..]), true, Kind::Float);
        let filtered =
            tch_utils::gabor::apply_gabor_filter(&gs, ANGLES_COUNT, 30, &FREQUENCIES, 0.45);

        let batch_size = patchs.size()[0];
        let filter_count = ANGLES_COUNT * FREQUENCIES.len();

        let masks_areas = masks.sum_dims([-3, -2, -1]);
        let mean = (&filtered * masks).sum_dims([-2, -1]) / &masks_areas.unsqueeze(-1);
        let variance = ((filtered - mean.view([batch_size, filter_count as i64, 1, 1])).square()
            * masks)
            .sum_dims([-2, -1])
            / masks_areas.unsqueeze(-1);

        let features = (0..filter_count).flat_map(|j| {
            let mean = mean.i((.., j as i64));
            let variance = variance.i((.., j as i64));
            let angle = (j / FREQUENCIES.len()) as f32 * 45.0;
            let frequency = FREQUENCIES[j % FREQUENCIES.len()] as f32;
            vec![
                Series::new(
                    &format!("gabor_angle_{}_frequency_{}_mean", angle, frequency),
                    Vec::<f32>::from(mean),
                ),
                Series::new(
                    &format!("gabor_angle_{}_frequency_{}_variance", angle, frequency),
                    Vec::<f32>::from(variance),
                ),
            ]
        });

        let features =
            std::iter::once(Series::new("centroid", centroids_to_key_strings(centroids)))
                .chain(features)
                .collect::<Vec<_>>();

        DataFrame::new(features).unwrap()
    }

    fn name(&self) -> &str {
        "gabor filter"
    }
}
