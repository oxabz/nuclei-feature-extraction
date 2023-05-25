use polars::{
    prelude::{DataFrame, NamedFrom},
    series::Series,
};
use tch::{index::*, Kind, Tensor};
use tch_utils::{glcm::{glcm}, glrlm::glrlm, tensor_ext::TensorExt};

use crate::utils::centroids_to_key_strings;

use super::FeatureSet;

const GLCM_LEVELS: [u8; 4] = [32, 64, 128, 255];
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
        let batch_size = patchs.size()[0];

        let mut features = Vec::with_capacity(14 * OFFSETS.len() * GLCM_LEVELS.len());

        let gs = patchs.mean_dim(Some(&([-3][..])), true, Kind::Float); // [N, 1, H, W]
        let glcms = OFFSETS
            .iter()
            .flat_map(|offset| {
                GLCM_LEVELS.iter()
                    .map(|level| {
                        (*offset, *level, glcm(&gs, *offset, *level, Some(masks), true))
                    })
            });
        for (offset, level, glcm) in glcms {
            // glcm: [N, LEVEL, LEVEL]
            let px = glcm.sum_dim_intlist(Some(&[-1][..]), false, Kind::Float); // [N, LEVEL]
            let py = glcm.sum_dim_intlist(Some(&[-2][..]), false, Kind::Float); // [N, LEVEL]

            let levels =
                Tensor::arange(level as i64, (Kind::Float, patchs.device())).unsqueeze(0); // [1, LEVEL]
            let intensity_x = (&levels * &px).sum_dim_intlist(Some(&[-1][..]), false, Kind::Float); // [N]
            let intensity_y = (&levels * &py).sum_dim_intlist(Some(&[-1][..]), false, Kind::Float); // [N]

            let (pxpy, pxdy) = {
                // [N, LEVEL * 2 - 2], [N, LEVEL]
                let pxpy = Tensor::zeros(
                    &[batch_size, level as i64 * 2 - 2],
                    (Kind::Float, patchs.device()),
                ); // [N, LEVEL * 2 - 2]
                let pxdy = Tensor::zeros(
                    &[batch_size, level as i64],
                    (Kind::Float, patchs.device()),
                ); // [N, LEVEL]
                for i in 0..level {
                    for j in 0..level {
                        let (i, j) = (i as i64, j as i64);
                        let idx1 = (i + j) - 2;
                        let idx2 = (i - j).abs();
                        let mut t1 = pxpy.i((.., idx1));
                        let mut t2 = pxdy.i((.., idx2));
                        t1 += glcm.i((.., i, j));
                        t2 += glcm.i((.., i, j));
                    }
                }
                (pxpy, pxdy)
            };

            let entropy_x =
                -(&px * (&px + 1e-6).log2()).sum_dim_intlist(Some(&[-1][..]), false, Kind::Float); // [N]
            let entropy_y =
                -(&py * (&py + 1e-6).log2()).sum_dim_intlist(Some(&[-1][..]), false, Kind::Float); // [N]
            let entropy_xy = -(&glcm * (&glcm + 1e-6).log2()).sum_dim_intlist(
                Some(&[-1, -2][..]),
                false,
                Kind::Float,
            ); // [N]

            let (hxy1, hxy2) = {
                // [N], [N]
                let pxpy = px.unsqueeze(-1).matmul(&py.unsqueeze(-2)); // [N, LEVEL, LEVEL]
                let hxy1 = -(&glcm * (&pxpy + 1e-6).log2()).sum_dim_intlist(
                    Some(&[-1, -2][..]),
                    false,
                    Kind::Float,
                );
                let hxy2 = -(&pxpy * (&pxpy + 1e-6).log2()).sum_dim_intlist(
                    Some(&[-1, -2][..]),
                    false,
                    Kind::Float,
                );
                (hxy1, hxy2)
            };

            let correlation =
                {
                    //[N]
                    let intensity =
                        Tensor::arange(level as i64, (Kind::Float, patchs.device()));
                    let intensity = intensity
                        .unsqueeze(1)
                        .matmul(&intensity.unsqueeze(0))
                        .unsqueeze(0); // [1, LEVEL, LEVEL]
                    (&glcm * intensity - (&intensity_x * &intensity_y).view([-1, 1, 1]))
                        .sum_dim_intlist(Some(&[-1, -2][..]), false, Kind::Float)
                };

            let (contrast, dissimilarity) =
                {
                    let i = Tensor::arange(level as i64, (Kind::Float, patchs.device()))
                        .view([1, 1, level as i64]);
                    let j = Tensor::arange(level as i64, (Kind::Float, patchs.device()))
                        .view([1, level as i64, 1]);
                    let imj = &i * (&i - &j).square(); // [1, LEVEL, LEVEL]
                    let contrast =
                        (&glcm * imj).sum_dim_intlist(Some(&[-1, -2][..]), false, Kind::Float); // [N]
                    let imj = (i - j).abs(); // [1, LEVEL, LEVEL]
                    let dissimilarity =
                        (&glcm * imj).sum_dim_intlist(Some(&[-1, -2][..]), false, Kind::Float);
                    (contrast, dissimilarity)
                };

            let entropy = -(&glcm * (&glcm + 1e-6).log2()).sum_dim_intlist(
                Some(&[-1, -2][..]),
                false,
                Kind::Float,
            );

            let angular_second_moment =
                glcm.square()
                    .sum_dim_intlist(Some(&[-1, -2][..]), false, Kind::Float);

            let (sum_average, sum_variance) = {
                // [N], [N]
                let k =
                    (Tensor::arange(level as i64 * 2 - 2, (Kind::Float, patchs.device()))
                        + 2)
                    .unsqueeze(0);
                let sum_avg = (&k * &pxpy).sum_dim_intlist(Some(&[-1][..]), false, Kind::Float); // [N]
                let sum_var = ((k - sum_avg.unsqueeze(1)).square() * &pxpy).sum_dim_intlist(
                    Some(&[-1][..]),
                    false,
                    Kind::Float,
                );
                (sum_avg, sum_var)
            };

            let sum_entropy = -(&pxpy * (&pxpy + 1e-6).log2()).sum_dim_intlist(
                Some(&[-1][..]),
                false,
                Kind::Float,
            );

            let sum_of_squares = {
                let i = Tensor::arange(level as i64, (Kind::Float, patchs.device()))
                    .view([1, level as i64]);
                let var = (i - intensity_x.unsqueeze(1)).square().unsqueeze(-1) * &glcm;
                var.sum_dim_intlist(Some(&[-1, -2][..]), false, Kind::Float)
            };

            let inverse_difference_moment = {
                let i = Tensor::arange(level as i64, (Kind::Float, patchs.device()))
                    .view([1, -1, 1]);
                let j = Tensor::arange(level as i64, (Kind::Float, patchs.device()))
                    .view([1, 1, -1]);
                let imj = (i - j).square();
                let idm = &glcm / (imj + 1.0);
                idm.sum_dim_intlist(Some(&[-1, -2][..]), false, Kind::Float)
            };

            let (difference_average, difference_variance) = {
                let k =
                    Tensor::arange(level as i64, (Kind::Float, patchs.device())).unsqueeze(0);
                let da = (&k * &pxdy).sum_dim_intlist(Some(&[-1][..]), false, Kind::Float); // [N]
                let dva = ((k - da.unsqueeze(-1)).square() * &pxdy).sum_dim_intlist(
                    Some(&[-1][..]),
                    false,
                    Kind::Float,
                );
                (da, dva)
            };

            let information_measure_correlation1 =
                (&entropy_xy - hxy1) / entropy_x.max_other(&entropy_y);

            let information_measure_correlation2 =
                (-((hxy2 - &entropy_xy) * -2.0).exp() + 1.0).sqrt();

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
                Vec::<f32>::from(information_measure_correlation1);
            let information_measure_correlation2 =
                Vec::<f32>::from(information_measure_correlation2);

            features.push((
                format!("correlation_{}_{}_{level}", offset.0, offset.1),
                correlation,
            ));
            features.push((format!("contrast_{}_{}_{level}", offset.0, offset.1), contrast));
            features.push((
                format!("dissimilarity_{}_{}_{level}", offset.0, offset.1),
                dissimilarity,
            ));
            features.push((format!("entropy_{}_{}_{level}", offset.0, offset.1), entropy));
            features.push((
                format!("angular_second_moment_{}_{}_{level}", offset.0, offset.1),
                angular_second_moment,
            ));
            features.push((
                format!("sum_average_{}_{}_{level}", offset.0, offset.1),
                sum_average,
            ));
            features.push((
                format!("sum_variance_{}_{}_{level}", offset.0, offset.1),
                sum_variance,
            ));
            features.push((
                format!("sum_entropy_{}_{}_{level}", offset.0, offset.1),
                sum_entropy,
            ));
            features.push((
                format!("sum_of_squares_{}_{}_{level}", offset.0, offset.1),
                sum_of_squares,
            ));
            features.push((
                format!("inverse_difference_moment_{}_{}_{level}", offset.0, offset.1),
                inverse_difference_moment,
            ));
            features.push((
                format!("difference_average_{}_{}_{level}", offset.0, offset.1),
                difference_average,
            ));
            features.push((
                format!("difference_variance_{}_{}_{level}", offset.0, offset.1),
                difference_variance,
            ));
            features.push((
                format!("information_measure_correlation1_{}_{}_{level}", offset.0, offset.1),
                information_measure_correlation1,
            ));
            features.push((
                format!("information_measure_correlation2_{}_{}_{level}", offset.0, offset.1),
                information_measure_correlation2,
            ));
        }

        let features =
            std::iter::once(Series::new("centroid", centroids_to_key_strings(centroids)))
                .chain(features.into_iter().map(|(k, v)| Series::new(&k, v)))
                .collect::<Vec<_>>();

        DataFrame::new(features).unwrap()
    }

    fn name(&self)->&str {
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

            // Number of runs [N]
            let nruns = glrlm.sum_dim_intlist(Some(&[-1, -2][..]), false, Kind::Float);

            // Gray Level Run-Length Vector [N, GLRLM_LEVELS]
            let glrlv = { glrlm.sum_dim_intlist(Some(&[-1][..]), false, Kind::Float) };

            // Run-length Run Number Vector [N, GLRLM_MAX_LENGTH]
            let rlrnv = { glrlm.sum_dim_intlist(Some(&[-2][..]), false, Kind::Float) };

            // Short Run Emphasis [N] & Long Run Emphasis [N]
            let (short_run_emphasis, long_run_emphasis) = {
                let j = Tensor::arange(GLRLM_MAX_LENGTH, (Kind::Float, patchs.device()))
                    .unsqueeze(0)
                    + 1.0;
                let j2 = j.square();
                let sre = &rlrnv / &j2;
                let lre = &rlrnv * j2;
                let sre = sre.sum_dim_intlist(Some(&[-1][..]), false, Kind::Float) / &nruns;
                let lre = lre.sum_dim_intlist(Some(&[-1][..]), false, Kind::Float) / &nruns;
                (sre, lre)
            };

            // Gray-Level Nonuniformity [N]
            let gray_level_nonuniformity = { glrlv.square().sum_dim(-1) / &nruns };

            // Run-Length Nonuniformity [N]
            let run_length_nonuniformity = { rlrnv.square().sum_dim(-1) / &nruns };

            // Run Percentage [N]
            let run_percentage = {
                let pix = masks.sum_dims([-3, -2, -1]);
                &nruns / pix
            };

            // Low Gray-Level Run Emphasis [N] & High Gray-Level Run Emphasis [N]
            let (low_gray_level_run_emphasis, high_gray_level_run_emphasis) = {
                let i = Tensor::arange(GLRLM_LEVELS as i64, (Kind::Float, patchs.device()))
                    .unsqueeze(0)
                    + 0.5;
                let lglre = &glrlv / i.square();
                let hglre = &glrlv * i.square();
                (lglre.sum_dim(-1) / &nruns, hglre.sum_dim(-1) / &nruns)
            };

            // Short Run Low Gray-Level Emphasis & Short Run High Gray-Level Emphasis & Long Run Low Gray-Level Emphasis & Long Run High Gray-Level Emphasis
            // Short Run Mid Gray-Level Empahsis & Long Run Mid Gray-Level Empahsis & Short Run Extreme Gray-Level Empahsis & Long Run Extreme Gray-Level Empahsis (I made it up)
            let (
                short_run_low_gray_level_emphasis,
                short_run_high_gray_level_emphasis,
                long_run_low_gray_level_emphasis,
                long_run_high_gray_level_emphasis,
                short_run_mid_gray_level_emphasis,
                long_run_mid_gray_level_emphasis,
                short_run_extreme_gray_level_emphasis,
                long_run_extreme_gray_level_emphasis,
            ) = {
                let j = Tensor::arange(GLRLM_MAX_LENGTH, (Kind::Float, patchs.device())) + 1.0;
                let j = j.view([1, 1, -1]);
                let j2 = j.square();
                let i = Tensor::arange(GLRLM_LEVELS as i64, (Kind::Float, patchs.device())) + 0.5;
                let i = i.view([1, -1, 1]);
                let i2 = i.square();
                let srlgle = &glrlm / (&i2 * &j2);
                let srhgle = &glrlm * &i2 / &j2;
                let lrlgle = &glrlm * &j2 / &i2;
                let lrhgle = &glrlm * i2 * &j2;
                let i = Tensor::arange(GLRLM_LEVELS as i64, (Kind::Float, patchs.device()))
                    - GLRLM_LEVELS as f64 / 2.0
                    + 0.5;
                let i = i.view([1, -1, 1]);
                let i2 = i.square();
                let srmgle = &glrlm / (&i2 * &j2);
                let lrmgle = &glrlm * &j2 / &i2;
                let srengle = &glrlm / (&i2 * &j2);
                let lrengle = &glrlm * &i2 * &j2;
                (
                    srlgle.sum_dims([-1, -2]) / &nruns,
                    srhgle.sum_dims([-1, -2]) / &nruns,
                    lrlgle.sum_dims([-1, -2]) / &nruns,
                    lrhgle.sum_dims([-1, -2]) / &nruns,
                    srmgle.sum_dims([-1, -2]) / &nruns,
                    lrmgle.sum_dims([-1, -2]) / &nruns,
                    srengle.sum_dims([-1, -2]) / &nruns,
                    lrengle.sum_dims([-1, -2]) / &nruns,
                )
            };

            let (run_length_mean, run_length_variance) = {
                let j = Tensor::arange(GLRLM_MAX_LENGTH, (Kind::Float, patchs.device())) + 1.0;
                let j = j.view([1, 1, -1]);
                let rlmean = (&glrlm * &j).mean_dim(Some(&[-2, -1][..]), false, Kind::Float);
                let rlvar = (glrlm * &j - &rlmean.view([-1, 1, 1])).square().mean_dim(
                    Some(&[-2, -1][..]),
                    false,
                    Kind::Float,
                );
                (rlmean, rlvar)
            };

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
                    Vec::<f32>::from(gray_level_nonuniformity),
                ),
                Series::new(
                    &format!("run_length_nonuniformity_{}_{}", direction.0, direction.1),
                    Vec::<f32>::from(run_length_nonuniformity),
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

    fn name(&self)->&str {
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

    fn name(&self)->&str {
        "gabor filter"
    }
}



