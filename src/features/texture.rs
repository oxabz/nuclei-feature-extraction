use tch::{Tensor, Kind, index::*};
use tch_utils::{glcm::glcm, glrlm::glrlm};

const GLCM_LEVELS: u8 = 64;

pub struct GlcmFeatures{
    pub correlation: f32,
    pub contraste: f32,
    pub dissimilarity: f32,
    pub entropy: f32,
    pub angular_second_moment: f32,
    pub sum_average: f32,
    pub sum_variance: f32,
    pub sum_entropy: f32,
    pub sum_of_squares: f32,
    pub inverse_difference_moment: f32,
    pub difference_variance: f32,
    pub information_measure_correlation1: f32,
    pub information_measure_correlation2: f32,
}

const OFFSETS: [(i64, i64);4] = [(0, 1), (1, 1), (1, 0), (1, -1)]; // 0, 45, 90, 135

pub fn glcm_features(image:&Tensor, mask:&Tensor)->Vec<GlcmFeatures>{
    let gs = image.mean_dim(Some(&([-3][..])), true, Kind::Float).unsqueeze(0); // [1, 1, H, W]
    let glcms = OFFSETS.iter()
        .map(|offset|glcm(&gs, *offset, GLCM_LEVELS, Some(mask)).squeeze())
        .collect::<Vec<_>>(); // [LEVEL, LEVEL] * 4
    let mut features = Vec::with_capacity(glcms.len());

    for glcm in glcms.iter() {
        // glcm: [LEVEL, LEVEL]
        let px = glcm.sum_dim_intlist(Some(&[-1][..]), false, Kind::Float); // [LEVEL]
        let py = glcm.sum_dim_intlist(Some(&[-2][..]), false, Kind::Float); // [LEVEL]

        let levels = Tensor::arange(GLCM_LEVELS as i64, (Kind::Float, image.device()));
        let intensity_x = (&levels * &px).sum(Kind::Float); // Scalar
        let intensity_y = (&levels * &py).sum(Kind::Float); // Scalar

        let std_x = px.std(true); // Scalar
        let std_y = py.std(true); // Scalar

        
        let (pxpy, pxdy) = { // [LEVEL * 2 - 2], [LEVEL]
            let pxpy = Tensor::zeros(&[GLCM_LEVELS as i64 * 2 - 2], (Kind::Float, image.device())); // [LEVEL * 2 - 2]
            let pxdy = Tensor::zeros(&[GLCM_LEVELS as i64], (Kind::Float, image.device())); // [LEVEL]
            for i in 0..GLCM_LEVELS {
                for j in 0..GLCM_LEVELS {
                    let (i, j) = (i as i64, j as i64);
                    let idx1 = (i + j) - 2;
                    let idx2 = (i - j).abs();
                    let mut t1 = pxpy.i(idx1);
                    let mut t2 = pxdy.i(idx2);
                    t1 += glcm.i((i,j));
                    t2 += glcm.i((i,j));
                }
            }
            (pxpy, pxdy)
        };

        let entropy_x = -(&px * (&px + 1e-6).log2()).sum(Kind::Float); // Scalar
        let entropy_y = -(&py * (&py + 1e-6).log2()).sum(Kind::Float); // Scalar
        let entropy_xy = -(glcm * (glcm + 1e-6).log2()).sum(Kind::Float); // Scalar

        let (hxy1, hxy2) = { // Scalar, Scalar
            let pxpy = px.unsqueeze(1).matmul(&py.unsqueeze(0));
            let hxy1 = - (glcm * (&pxpy + 1e-6).log2()).sum(Kind::Float);
            let hxy2 = - (&pxpy * (&pxpy + 1e-6).log2()).sum(Kind::Float);
            (hxy1, hxy2)
        };

        let correlation = {
            let intensity = Tensor::arange(GLCM_LEVELS as i64, (Kind::Float, image.device()));
            let intensity = intensity.unsqueeze(1).matmul(&intensity.unsqueeze(0));
            (glcm * intensity - &intensity_x * &intensity_y).sum(Kind::Float)
        };
        let correlation = f32::from(correlation);

        let (contraste, dissimilarity) = {
            let i = Tensor::arange(GLCM_LEVELS as i64, (Kind::Float, image.device())).unsqueeze(1).repeat(&[1,GLCM_LEVELS as i64]);
            let j = Tensor::arange(GLCM_LEVELS as i64, (Kind::Float, image.device())).unsqueeze(0).repeat(&[GLCM_LEVELS as i64,1]);
            let imj = &i * (&i - &j).square();
            let contrast = (glcm * imj).sum(Kind::Float);
            let imj = (i - j).abs();
            let dissimilarity = (glcm * imj).sum(Kind::Float);
            (f32::from(contrast), f32::from(dissimilarity))
        };

        let entropy = f32::from(- (glcm * (glcm + 1e-6).log2()).sum(Kind::Float));

        let angular_second_moment = f32::from(glcm.square().sum(Kind::Float));

        let sum_average = {
            let k = Tensor::arange(GLCM_LEVELS as i64 * 2 - 2, (Kind::Float, image.device())) + 2;
            let sum = (k * &pxpy).sum(Kind::Float);
            f32::from(sum)
        };

        let sum_variance = {
            let k = Tensor::arange(GLCM_LEVELS as i64 * 2 - 2, (Kind::Float, image.device())) + 2;
            let sum = (- sum_average + k).square() * &pxpy;
            f32::from(sum.sum(Kind::Float))
        };

        let sum_entropy = {
            let sum = &pxpy * (&pxpy + 1e-6).log2();
            f32::from(- sum.sum(Kind::Float))
        };

        let sum_of_squares = {
            let i = Tensor::arange(GLCM_LEVELS as i64, (Kind::Float, image.device())).unsqueeze(1).repeat(&[1, GLCM_LEVELS as i64]);
            let var = (i - &intensity_x).square() * glcm;
            f32::from(var.sum(Kind::Float))
        };

        let inverse_difference_moment = {
            let i = Tensor::arange(GLCM_LEVELS as i64, (Kind::Float, image.device())).unsqueeze(1).repeat(&[1, GLCM_LEVELS as i64]);
            let j = Tensor::arange(GLCM_LEVELS as i64, (Kind::Float, image.device())).unsqueeze(0).repeat(&[GLCM_LEVELS as i64, 1]);
            let imj = (i - j).square();
            let idm = glcm / (imj + 1.0);
            f32::from(idm.sum(Kind::Float))
        };

        let difference_variance = {
            let k = Tensor::arange(GLCM_LEVELS as i64, (Kind::Float, image.device()));
            let da = (&k * &pxdy).sum(Kind::Float);
            let dva = (- da + &k).square() * &pxdy;
            f32::from(dva.sum(Kind::Float))
        };

        let information_measure_correlation1 = {
            let imc1 = &entropy_xy - hxy1 / entropy_x.max_other(&entropy_y);
            f32::from(imc1)
        };

        let information_measure_correlation2 = {
            let imc2 = ( - ((hxy2 - &entropy_xy) * -2.0).exp() + 1.0).sqrt();
            f32::from(imc2)
        };
        
        features.push(GlcmFeatures{
            correlation,
            contraste,
            dissimilarity,
            entropy,
            angular_second_moment,
            sum_average,
            sum_variance,
            sum_entropy,
            sum_of_squares,
            inverse_difference_moment,
            difference_variance,
            information_measure_correlation1,
            information_measure_correlation2,
        });
    }
    features
}

const GLRLM_LEVELS: u8 = 24;
const GLRLM_MAX_LENGTH: i64 = 16;

const DIRECTIONS: [(i64, i64); 4] = [(1,0), (1,1), (0,1), (-1, 1)];

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
    pub run_length_variance: f32,
}

pub fn glrlm_features(patch: &Tensor, mask:&Tensor) -> Vec<GLRLMFeatures>{
    let gs = patch.mean_dim(Some(&[-3][..]), true, Kind::Float);
    let mut glrlms = Vec::with_capacity(DIRECTIONS.len());
    
    for direction in &DIRECTIONS{
        let glrlm = glrlm(&gs.unsqueeze(0), GLRLM_LEVELS, GLRLM_MAX_LENGTH, *direction, Some(mask)).squeeze().to_kind(Kind::Float);
        
        // Number of runs
        let nruns = glrlm.sum(Kind::Float);

        // Gray Level Run-Length Pixel Number Matrix
        let glrlpnm = {
            let j = Tensor::arange(GLRLM_MAX_LENGTH, (Kind::Float, patch.device())).unsqueeze(0).repeat(&[GLRLM_LEVELS as i64, 1]) + 1.0;
            &glrlm * j
        };

        // Gray Level Run-Length Vector
        let glrnv = {
            glrlm.sum_dim_intlist(Some(&[1][..]), false, Kind::Float)
        };

        // Run-length Run Number Vector
        let rlrnv = {
            glrlm.sum_dim_intlist(Some(&[0][..]), false, Kind::Float)
        };

        // Gray-Level Run-Length-One Vector
        let glrl1v = glrlm.i((.., 0));

        // Short Run Emphasis
        let short_run_emphasis = {
            let j = Tensor::arange(GLRLM_MAX_LENGTH, (Kind::Float, patch.device())) + 1.0;
            let sre = &rlrnv / j.square();
            f32::from(sre.sum(Kind::Float) / &nruns)
        };

        // Long Run Emphasis
        let long_run_emphasis = {
            let j = Tensor::arange(GLRLM_MAX_LENGTH, (Kind::Float, patch.device())) + 1.0;
            let lre = &rlrnv * j.square();
            f32::from(lre.sum(Kind::Float) / &nruns)
        };

        // Gray-Level Nonuniformity
        let gray_level_nonuniformity = {
            f32::from(glrlm.square().sum(Kind::Float) / &nruns)
        };

        // Run-Length Nonuniformity
        let run_length_nonuniformity = {
            f32::from(rlrnv.square().sum(Kind::Float) / &nruns)
        };

        // Run Percentage
        let run_percentage = {
            let (height, width) = glrlm.size2().unwrap();
        
            f32::from(&nruns / (height as f64 * width as f64))
        };

        // Low Gray-Level Run Emphasis
        let low_gray_level_run_emphasis = {
            let i = Tensor::arange(GLRLM_LEVELS as i64, (Kind::Float, patch.device()));
            let lglre = &glrnv / i.square();
            f32::from(lglre.sum(Kind::Float) / &nruns)
        };

        // High Gray-Level Run Emphasis
        let high_gray_level_run_emphasis = {
            let i = Tensor::arange(GLRLM_LEVELS as i64, (Kind::Float, patch.device()));
            let hglre = &glrnv * i.square();
            f32::from(hglre.sum(Kind::Float) / &nruns)
        };

        // Short Run Low Gray-Level Emphasis & Short Run High Gray-Level Emphasis & Long Run Low Gray-Level Emphasis & Long Run High Gray-Level Emphasis
        let (
            short_run_low_gray_level_emphasis,
            short_run_high_gray_level_emphasis,
            long_run_low_gray_level_emphasis,
            long_run_high_gray_level_emphasis,
            ) = {
            let j = Tensor::arange(GLRLM_MAX_LENGTH, (Kind::Float, patch.device())) + 1.0;
            let j = j.unsqueeze(0).repeat(&[GLRLM_LEVELS as i64, 1]);
            let j2 = j.square();
            let i = Tensor::arange(GLRLM_LEVELS as i64, (Kind::Float, patch.device()));
            let i = i.unsqueeze(1).repeat(&[1, GLRLM_MAX_LENGTH as i64]);
            let i2 = i.square();
            let srlgle = &glrlm / (&i2 * &j2);
            let srhgle = &glrlm * &i2 / &j2;
            let lrlgle = &glrlm * &j2 / &i2;
            let lrhgle = &glrlm * i2 * j2;
            (
                f32::from(srlgle.sum(Kind::Float) / &nruns),
                f32::from(srhgle.sum(Kind::Float) / &nruns),
                f32::from(lrlgle.sum(Kind::Float) / &nruns),
                f32::from(lrhgle.sum(Kind::Float) / &nruns),
            )
        };

        // Short Run Mid Gray-Level Empahsis & Long Run Mid Gray-Level Empahsis & Short Run Extreme Gray-Level Empahsis & Long Run Extreme Gray-Level Empahsis
        // source : I made it up
        let (
            short_run_mid_gray_level_emphasis,
            long_run_mid_gray_level_emphasis,
            short_run_extreme_gray_level_emphasis,
            long_run_extreme_gray_level_emphasis,
            ) = {
            let j = Tensor::arange(GLRLM_MAX_LENGTH, (Kind::Float, patch.device())) + 1.0;
            let j = j.unsqueeze(0).repeat(&[GLRLM_LEVELS as i64, 1]);
            let j2 = j.square();
            let i = Tensor::arange(GLRLM_LEVELS as i64, (Kind::Float, patch.device())) - GLRLM_LEVELS as f64 / 2.0;
            let i = i.unsqueeze(1).repeat(&[1, GLRLM_MAX_LENGTH as i64]);
            let i2 = i.square();
            let srmgle = &glrlm / (&i2 * &j2);
            let lrmgle = &glrlm * &j2 / &i2;
            let srengle = &glrlm / (&i2 * &j2);
            let lrengle = &glrlm * &i2 * &j2;
            (
                f32::from(srmgle.sum(Kind::Float) / &nruns),
                f32::from(lrmgle.sum(Kind::Float) / &nruns),
                f32::from(srengle.sum(Kind::Float) / &nruns),
                f32::from(lrengle.sum(Kind::Float) / nruns),
            )
        };

        let (_, run_length_variance) = {
            let j = Tensor::arange(GLRLM_MAX_LENGTH, (Kind::Float, patch.device())) + 1.0;
            let j = j.unsqueeze(0).repeat(&[GLRLM_LEVELS as i64, 1]);
            let rlmean = (&glrlm * &j).mean(Kind::Float);
            let rlvar = (glrlm * &j - &rlmean).square().mean(Kind::Float);
            (f32::from(rlmean), f32::from(rlvar.sum(Kind::Float)))
        };

        glrlms.push(
            GLRLMFeatures{
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
                run_length_variance,
            }
        )
    }

    glrlms
}
