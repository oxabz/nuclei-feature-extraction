use struct_field_names_as_array::FieldNamesAsArray;
use tch::{Tensor, Kind, index::*};
use tch_utils::{glcm::glcm, glrlm::glrlm, tensor_ext::TensorExt};

const GLCM_LEVELS: u8 = 64;

#[derive(Debug, FieldNamesAsArray)]
pub struct GlcmFeatures{
    pub correlation: f32,
    pub contrast: f32,
    pub dissimilarity: f32,
    pub entropy: f32,
    pub angular_second_moment: f32,
    pub sum_average: f32,
    pub sum_variance: f32,
    pub sum_entropy: f32,
    pub sum_of_squares: f32,
    pub inverse_difference_moment: f32,
    pub difference_average: f32,
    pub difference_variance: f32,
    pub information_measure_correlation1: f32,
    pub information_measure_correlation2: f32,
}

impl GlcmFeatures {
    pub fn write_header_to_csv<W: std::io::Write>(writer: &mut csv::Writer<W>) -> Result<(), csv::Error> {
        let headers = OFFSETS.iter()
            .flat_map(|offset|{
                let (dx, dy) = offset;
                GlcmFeatures::FIELD_NAMES_AS_ARRAY.iter()
                    .map(move |field_name|format!("{field_name}_{dx}_{dy}"))
            })
            .collect::<Vec<_>>();
        let mut rec = vec!["centroid_x".to_string(), "centroid_y".to_string()];
        rec.extend(headers);
        writer.write_record(rec)?;
        Ok(())
    }
    
    pub fn as_slice(&self)->&[f32]{
        unsafe{
            std::slice::from_raw_parts(
                &self.correlation as *const f32,
                std::mem::size_of::<GlcmFeatures>() / std::mem::size_of::<f32>()
            )
        }
    }
}

const OFFSETS: [(i64, i64);4] = [(0, 1), (1, 1), (1, 0), (1, -1)]; // 0, 45, 90, 135

pub fn glcm_features(patches: &Tensor, masks:&Tensor)->(Vec<Vec<GlcmFeatures>>, Vec<(i64, i64)>){
    let _ = tch::no_grad_guard();
    let batch_size = patches.size()[0];

    let gs = patches.mean_dim(Some(&([-3][..])), true, Kind::Float); // [N, 1, H, W]
    let glcms = OFFSETS.iter()
        .map(|offset| glcm(&gs, *offset, GLCM_LEVELS, Some(masks)))
        .collect::<Vec<_>>(); // [N, LEVEL, LEVEL] * 4
    let mut features = Vec::with_capacity(batch_size as usize);
    for _ in 0..batch_size {
        features.push(Vec::with_capacity(glcms.len()))
    }

    for glcm in glcms.iter() {
        // glcm: [N, LEVEL, LEVEL]
        let px = glcm.sum_dim_intlist(Some(&[-1][..]), false, Kind::Float); // [N, LEVEL]
        let py = glcm.sum_dim_intlist(Some(&[-2][..]), false, Kind::Float); // [N, LEVEL]

        let levels = Tensor::arange(GLCM_LEVELS as i64, (Kind::Float, patches.device())).unsqueeze(0); // [1, LEVEL]
        let intensity_x = (&levels * &px).sum_dim_intlist(Some(&[-1][..]), false, Kind::Float); // [N]
        let intensity_y = (&levels * &py).sum_dim_intlist(Some(&[-1][..]), false, Kind::Float); // [N]

        let std_x = px.std_dim(Some(&[-1][..]), true, false); // [N]
        let std_y = py.std_dim(Some(&[-1][..]), true, false); // [N]

        
        let (pxpy, pxdy) = { // [N, LEVEL * 2 - 2], [N, LEVEL]
            let pxpy = Tensor::zeros(&[batch_size, GLCM_LEVELS as i64 * 2 - 2], (Kind::Float, patches.device())); // [N, LEVEL * 2 - 2]
            let pxdy = Tensor::zeros(&[batch_size, GLCM_LEVELS as i64], (Kind::Float, patches.device())); // [N, LEVEL]
            for i in 0..GLCM_LEVELS {
                for j in 0..GLCM_LEVELS {
                    let (i, j) = (i as i64, j as i64);
                    let idx1 = (i + j) - 2;
                    let idx2 = (i - j).abs();
                    let mut t1 = pxpy.i((.., idx1));
                    let mut t2 = pxdy.i((.., idx2));
                    t1 += glcm.i((..,i,j));
                    t2 += glcm.i((..,i,j));
                }
            }
            (pxpy, pxdy)
        };

        let entropy_x = -(&px * (&px + 1e-6).log2()).sum_dim_intlist(Some(&[-1][..]), false, Kind::Float); // [N]
        let entropy_y = -(&py * (&py + 1e-6).log2()).sum_dim_intlist(Some(&[-1][..]), false, Kind::Float); // [N]
        let entropy_xy = -(glcm * (glcm + 1e-6).log2()).sum_dim_intlist(Some(&[-1, -2][..]), false, Kind::Float); // [N]

        let (hxy1, hxy2) = { // [N], [N]
            let pxpy = px.unsqueeze(-1).matmul(&py.unsqueeze(-2)); // [N, LEVEL, LEVEL]
            let hxy1 = - (glcm * (&pxpy + 1e-6).log2()).sum_dim_intlist(Some(&[-1, -2][..]), false, Kind::Float);
            let hxy2 = - (&pxpy * (&pxpy + 1e-6).log2()).sum_dim_intlist(Some(&[-1, -2][..]), false, Kind::Float);
            (hxy1, hxy2)
        };

        let correlation = { //[N]
            let intensity = Tensor::arange(GLCM_LEVELS as i64, (Kind::Float, patches.device()));
            let intensity = intensity.unsqueeze(1).matmul(&intensity.unsqueeze(0)).unsqueeze(0); // [1, LEVEL, LEVEL]
            (glcm * intensity - &intensity_x * &intensity_y).sum_dim_intlist(Some(&[-1, -2][..]), false, Kind::Float)
        };

        let (contrast, dissimilarity) = {
            let i = Tensor::arange(GLCM_LEVELS as i64, (Kind::Float, patches.device())).view([1, 1, GLCM_LEVELS as i64]);
            let j = Tensor::arange(GLCM_LEVELS as i64, (Kind::Float, patches.device())).view([1, GLCM_LEVELS as i64, 1]);
            let imj = &i * (&i - &j).square(); // [1, LEVEL, LEVEL]
            let contrast = (glcm * imj).sum_dim_intlist(Some(&[-1, -2][..]), false, Kind::Float); // [N]
            let imj = (i - j).abs(); // [1, LEVEL, LEVEL]
            let dissimilarity = (glcm * imj).sum_dim_intlist(Some(&[-1, -2][..]), false, Kind::Float);
            (contrast, dissimilarity)
        };

        let entropy = - (glcm * (glcm + 1e-6).log2()).sum_dim_intlist(Some(&[-1, -2][..]), false, Kind::Float);

        let angular_second_moment = glcm.square().sum_dim_intlist(Some(&[-1, -2][..]), false, Kind::Float);

        let (sum_average, sum_variance) = { // [N], [N]
            let k = (Tensor::arange(GLCM_LEVELS as i64 * 2 - 2, (Kind::Float, patches.device())) + 2).unsqueeze(0);
            let sum_avg = (&k * &pxpy).sum_dim_intlist(Some(&[-1][..]), false, Kind::Float); // [N]
            let sum_var = ((k - sum_avg.unsqueeze(1)).square() * &pxpy).sum_dim_intlist(Some(&[-1][..]), false, Kind::Float);
            (sum_avg, sum_var)
        };

        let sum_entropy = - (&pxpy * (&pxpy + 1e-6).log2()).sum_dim_intlist(Some(&[-1][..]), false, Kind::Float);

        let sum_of_squares = {
            let i = Tensor::arange(GLCM_LEVELS as i64, (Kind::Float, patches.device())).view([1, GLCM_LEVELS as i64]);
            let var = (i - intensity_x.unsqueeze(1)).square().unsqueeze(-1) * glcm;
            var.sum_dim_intlist(Some(&[-1,-2][..]), false, Kind::Float)
        };

        let inverse_difference_moment = {
            let i = Tensor::arange(GLCM_LEVELS as i64, (Kind::Float, patches.device())).view([1,-1,1]);
            let j = Tensor::arange(GLCM_LEVELS as i64, (Kind::Float, patches.device())).view([1,1,-1]);
            let imj = (i - j).square();
            let idm = glcm / (imj + 1.0);
            idm.sum_dim_intlist(Some(&[-1,-2][..]), false, Kind::Float)
        };

        let (difference_average, difference_variance) = {
            let k = Tensor::arange(GLCM_LEVELS as i64, (Kind::Float, patches.device())).unsqueeze(0);
            let da = (&k * &pxdy).sum_dim_intlist(Some(&[-1][..]), false, Kind::Float); // [N]
            let dva = ((k - da.unsqueeze(-1)).square() * &pxdy).sum_dim_intlist(Some(&[-1][..]), false, Kind::Float);
            (da, dva)
        };

        let information_measure_correlation1 = (&entropy_xy - hxy1) / entropy_x.max_other(&entropy_y);

        let information_measure_correlation2 = (- ((hxy2 - &entropy_xy) * -2.0).exp() + 1.0).sqrt();
        
        for i in 0..batch_size{
            let correlation = f32::from(correlation.i(i));
            let contrast = f32::from(contrast.i(i));
            let dissimilarity = f32::from(dissimilarity.i(i));
            let entropy = f32::from(entropy.i(i));
            let angular_second_moment = f32::from(angular_second_moment.i(i));
            let sum_average = f32::from(sum_average.i(i));
            let sum_variance = f32::from(sum_variance.i(i));
            let sum_entropy = f32::from(sum_entropy.i(i));
            let sum_of_squares = f32::from(sum_of_squares.i(i));
            let inverse_difference_moment = f32::from(inverse_difference_moment.i(i));
            let difference_average = f32::from(difference_average.i(i));
            let difference_variance = f32::from(difference_variance.i(i));
            let information_measure_correlation1 = f32::from(information_measure_correlation1.i(i));
            let information_measure_correlation2 = f32::from(information_measure_correlation2.i(i));

            features[i as usize].push(GlcmFeatures{
                correlation,
                contrast,
                dissimilarity,
                entropy,
                angular_second_moment,
                sum_average,
                sum_variance,
                sum_entropy,
                sum_of_squares,
                inverse_difference_moment,
                difference_average,
                difference_variance,
                information_measure_correlation1,
                information_measure_correlation2,
            });
        }
    }
    (features, OFFSETS.to_vec())
}

const GLRLM_LEVELS: u8 = 24;
const GLRLM_MAX_LENGTH: i64 = 16;
const DIRECTIONS: [(i64, i64); 4] = [(1,0), (1,1), (0,1), (-1, 1)];

#[derive(Debug, Clone, FieldNamesAsArray)]
pub struct GLRLMFeatures{
    pub short_run_emphasis: f32,
    pub long_run_emphasis: f32,
    pub gray_level_nonuniformity: f32,
    pub run_length_nonuniformity: f32,
    pub low_gray_level_run_emphasis: f32,
    pub high_gray_level_run_emphasis: f32,
    pub short_run_low_gray_level_emphasis: f32,
    pub short_run_high_gray_level_emphasis: f32,
    pub long_run_low_gray_level_emphasis: f32,
    pub long_run_high_gray_level_emphasis: f32,
    pub short_run_mid_gray_level_emphasis: f32,
    pub long_run_mid_gray_level_emphasis: f32,
    pub short_run_extreme_gray_level_emphasis: f32,
    pub long_run_extreme_gray_level_emphasis: f32,
    pub run_percentage: f32,
    pub run_length_mean: f32,
    pub run_length_variance: f32,
}

impl GLRLMFeatures{
    pub fn write_header_to_csv<W: std::io::Write>(writer: &mut csv::Writer<W>) -> Result<(), csv::Error> {
        let headers = OFFSETS.iter()
            .flat_map(|offset|{
                let (dx, dy) = offset;
                GlcmFeatures::FIELD_NAMES_AS_ARRAY.iter()
                    .map(move |field_name|format!("{field_name}_{dx}_{dy}"))
            })
            .collect::<Vec<_>>();
        let mut rec = vec!["centroid_x".to_string(), "centroid_y".to_string()];
        rec.extend(headers);
        writer.write_record(rec)?;
        Ok(())
    }
    
    pub fn as_slice(&self)->&[f32]{
        unsafe{
            std::slice::from_raw_parts(
                &self.short_run_emphasis as *const f32,
                std::mem::size_of::<GLRLMFeatures>() / std::mem::size_of::<f32>()
            )
        }
    }
}

pub fn glrlm_features(patches: &Tensor, masks:&Tensor) -> Vec<Vec<GLRLMFeatures>>{
    let batch_size = patches.size()[0];

    let gs = patches.mean_dim(Some(&[-3][..]), true, Kind::Float);
    let mut glrlms = Vec::with_capacity(batch_size as usize);

    for _ in 0..batch_size{
        glrlms.push(Vec::with_capacity(DIRECTIONS.len()));
    }
    
    for direction in &DIRECTIONS{
        // Gray Level Run-Length Matrix [N, GLRLM_LEVELS, GLRLM_MAX_LENGTH]
        let glrlm = glrlm(&gs, GLRLM_LEVELS, GLRLM_MAX_LENGTH, *direction, Some(masks)).to_kind(Kind::Float);
        
        // Number of runs [N]
        let nruns = glrlm.sum_dim_intlist(Some(&[-1,-2][..]), false, Kind::Float);

        // Gray Level Run-Length Pixel Number Matrix [N, GLRLM_LEVELS, GLRLM_MAX_LENGTH]
        let glrlpnm = {
            let j = Tensor::arange(GLRLM_MAX_LENGTH, (Kind::Float, patches.device())).view([1,1,-1]) + 1.0;
            &glrlm * j
        };

        // Gray Level Run-Length Vector [N, GLRLM_LEVELS]
        let glrlv = {
            glrlm.sum_dim_intlist(Some(&[-1][..]), false, Kind::Float)
        };

        // Run-length Run Number Vector [N, GLRLM_MAX_LENGTH]
        let rlrnv = {
            glrlm.sum_dim_intlist(Some(&[-2][..]), false, Kind::Float)
        };

        // Gray-Level Run-Length-One Vector [N, GLRLM_LEVELS]
        let glrl1v = glrlm.select(-1, 0);

        // Short Run Emphasis [N] & Long Run Emphasis [N]
        let (short_run_emphasis, long_run_emphasis) = {
            let j = Tensor::arange(GLRLM_MAX_LENGTH, (Kind::Float, patches.device())).unsqueeze(0) + 1.0;
            let j2 = j.square();
            let sre = &rlrnv / &j2;
            let lre = &rlrnv * j2;
            let sre = sre.sum_dim_intlist(Some(&[-1][..]), false, Kind::Float) / &nruns;
            let lre = lre.sum_dim_intlist(Some(&[-1][..]), false, Kind::Float) / &nruns;
            (sre, lre)
        };

        // Gray-Level Nonuniformity [N]
        let gray_level_nonuniformity = {
            glrlv.square().sum_dim(-1) / &nruns
        };

        // Run-Length Nonuniformity [N]
        let run_length_nonuniformity = {
            rlrnv.square().sum_dim(-1) / &nruns
        };

        // Run Percentage [N]
        let run_percentage = {
            let pix = masks.sum_dims([-3,-2,-1]);
            &nruns / pix
        };

        // Low Gray-Level Run Emphasis [N] & High Gray-Level Run Emphasis [N]
        let (low_gray_level_run_emphasis, high_gray_level_run_emphasis) = {
            let i = Tensor::arange(GLRLM_LEVELS as i64, (Kind::Float, patches.device())).unsqueeze(0);
            let lglre = &glrlv / i.square();
            let hglre = &glrlv * i.square();
            (
                lglre.sum_dim(-1) / &nruns,
                hglre.sum_dim(-1) / &nruns,
            )
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
            let j = Tensor::arange(GLRLM_MAX_LENGTH, (Kind::Float, patches.device())) + 1.0;
            let j = j.view([1,1,-1]);
            let j2 = j.square();
            let i = Tensor::arange(GLRLM_LEVELS as i64, (Kind::Float, patches.device()));
            let i = i.view([1,-1,1]);
            let i2 = i.square();
            let srlgle = &glrlm / (&i2 * &j2);
            let srhgle = &glrlm * &i2 / &j2;
            let lrlgle = &glrlm * &j2 / &i2;
            let lrhgle = &glrlm * i2 * &j2;
            let i = Tensor::arange(GLRLM_LEVELS as i64, (Kind::Float, patches.device())) - GLRLM_LEVELS as f64 / 2.0;
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
            let j = Tensor::arange(GLRLM_MAX_LENGTH, (Kind::Float, patches.device())) + 1.0;
            let j = j.view([1,1,-1]);
            let rlmean = (&glrlm * &j).mean_dim(Some(&[-2,-1][..]), false, Kind::Float);
            let rlvar = (glrlm * &j - &rlmean.view([-1, 1, 1])).square().mean_dim(Some(&[-2,-1][..]), false, Kind::Float);
            (rlmean, rlvar)
        };

        for i in 0..batch_size{
            let short_run_emphasis = f32::from(short_run_emphasis.i(i));
            let long_run_emphasis = f32::from(long_run_emphasis.i(i));
            let gray_level_nonuniformity = f32::from(gray_level_nonuniformity.i(i));
            let run_length_nonuniformity = f32::from(run_length_nonuniformity.i(i));
            let low_gray_level_run_emphasis = f32::from(low_gray_level_run_emphasis.i(i));
            let high_gray_level_run_emphasis = f32::from(high_gray_level_run_emphasis.i(i));
            let short_run_low_gray_level_emphasis = f32::from(short_run_low_gray_level_emphasis.i(i));
            let short_run_high_gray_level_emphasis = f32::from(short_run_high_gray_level_emphasis.i(i));
            let long_run_low_gray_level_emphasis = f32::from(long_run_low_gray_level_emphasis.i(i));
            let long_run_high_gray_level_emphasis = f32::from(long_run_high_gray_level_emphasis.i(i));
            let short_run_mid_gray_level_emphasis = f32::from(short_run_mid_gray_level_emphasis.i(i));
            let long_run_mid_gray_level_emphasis = f32::from(long_run_mid_gray_level_emphasis.i(i));
            let short_run_extreme_gray_level_emphasis = f32::from(short_run_extreme_gray_level_emphasis.i(i));
            let long_run_extreme_gray_level_emphasis = f32::from(long_run_extreme_gray_level_emphasis.i(i));
            let run_percentage = f32::from(run_percentage.i(i));
            let run_length_mean = f32::from(run_length_mean.i(i));
            let run_length_variance = f32::from(run_length_variance.i(i));

            glrlms[i as usize].push(
                GLRLMFeatures {
                    short_run_emphasis,
                    long_run_emphasis,
                    gray_level_nonuniformity,
                    run_length_nonuniformity,
                    low_gray_level_run_emphasis,
                    high_gray_level_run_emphasis,
                    short_run_low_gray_level_emphasis,
                    short_run_high_gray_level_emphasis,
                    long_run_low_gray_level_emphasis,
                    long_run_high_gray_level_emphasis,
                    short_run_mid_gray_level_emphasis,
                    long_run_mid_gray_level_emphasis,
                    short_run_extreme_gray_level_emphasis,
                    long_run_extreme_gray_level_emphasis,
                    run_percentage,
                    run_length_mean,
                    run_length_variance,
                }
            )
        }
    }

    glrlms
}
