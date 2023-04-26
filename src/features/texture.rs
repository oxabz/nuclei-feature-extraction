use tch::{Tensor, Kind, index::*};
use tch_utils::glcm::glcm;

const LEVELS: u8 = 254;

struct GlcmFeatures{
    correlation: f32,
    contraste: f32,
    dissimilarity: f32,
    entropy: f32,
    angular_second_moment: f32,
    sum_average: f32,
    sum_variance: f32,
    sum_entropy: f32,
    sum_of_squares: f32,
    inverse_difference_moment: f32,
    difference_variance: f32,
    information_measure_correlation1: f32,
    information_measure_correlation2: f32,
}

const OFFSETS: [(i64, i64);4] = [(0, 1), (1, 1), (1, 0), (1, -1)]; // 0, 45, 90, 135

fn glcm_features(image:&Tensor, mask:&Tensor)->Vec<GlcmFeatures>{
    let gs = image.mean_dim(Some(&([-3][..])), true, Kind::Float).unsqueeze(0); // [1, 1, H, W]
    let glcms = OFFSETS.iter()
        .map(|offset|glcm(&gs, *offset, LEVELS, Some(mask)).squeeze())
        .collect::<Vec<_>>(); // [LEVEL, LEVEL] * 4
    
    let mut features = Vec::with_capacity(glcms.len());

    for glcm in glcms.iter() {
        // glcm: [LEVEL, LEVEL]
        let px = glcm.sum_dim_intlist(Some(&[-1][..]), false, Kind::Float); // [LEVEL]
        let py = glcm.sum_dim_intlist(Some(&[-2][..]), false, Kind::Float); // [LEVEL]

        let levels = Tensor::arange(LEVELS as i64, (Kind::Float, image.device()));
        let intensity_x = (&levels * &px).sum(Kind::Float); // Scalar
        let intensity_y = (&levels * &py).sum(Kind::Float); // Scalar

        let std_x = px.std(true); // Scalar
        let std_y = py.std(true); // Scalar

        
        let (pxpy, pxdy) = { // [LEVEL * 2 - 2], [LEVEL]
            let pxpy = Tensor::zeros(&[LEVELS as i64 * 2 - 2], (Kind::Float, image.device())); // [LEVEL * 2 - 2]
            let pxdy = Tensor::zeros(&[LEVELS as i64], (Kind::Float, image.device())); // [LEVEL]
            for i in 0..LEVELS {
                for j in 0..LEVELS {
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
            let intensity = Tensor::arange(LEVELS as i64, (Kind::Float, image.device()));
            let intensity = intensity.unsqueeze(1).matmul(&intensity.unsqueeze(0));
            (glcm * intensity - &intensity_x * &intensity_y).sum(Kind::Float)
        };
        let correlation = f32::from(correlation);

        let (contraste, dissimilarity) = {
            let i = Tensor::arange(LEVELS as i64, (Kind::Float, image.device())).unsqueeze(1).repeat(&[LEVELS as i64,1]);
            let j = Tensor::arange(LEVELS as i64, (Kind::Float, image.device())).unsqueeze(0).repeat(&[1,LEVELS as i64]);
            let imj = &i * (&i - &j).square();
            let contrast = (glcm * imj).sum(Kind::Float);
            let imj = (i - j).abs();
            let dissimilarity = (glcm * imj).sum(Kind::Float);
            (f32::from(contrast), f32::from(dissimilarity))
        };

        let entropy = f32::from(- (glcm * (glcm + 1e-6).log2()).sum(Kind::Float));

        let angular_second_moment = f32::from(glcm.square().sum(Kind::Float));

        let sum_average = {
            let k = Tensor::arange(LEVELS as i64 * 2 - 2, (Kind::Float, image.device())) + 2;
            let sum = (k * &pxpy).sum(Kind::Float);
            f32::from(sum)
        };

        let sum_variance = {
            let k = Tensor::arange(LEVELS as i64 * 2 - 2, (Kind::Float, image.device())) + 2;
            let sum = (- sum_average + k).square() * &pxpy;
            f32::from(sum.sum(Kind::Float))
        };

        let sum_entropy = {
            let sum = &pxpy * (&pxpy + 1e-6).log2();
            f32::from(- sum.sum(Kind::Float))
        };

        let sum_of_squares = {
            let i = Tensor::arange(LEVELS as i64, (Kind::Float, image.device())).unsqueeze(1).repeat(&[LEVELS as i64, 1]);
            let var = (i - &intensity_x).square() * glcm;
            f32::from(var.sum(Kind::Float))
        };

        let inverse_difference_moment = {
            let i = Tensor::arange(LEVELS as i64, (Kind::Float, image.device())).unsqueeze(1).repeat(&[LEVELS as i64, 1]);
            let j = Tensor::arange(LEVELS as i64, (Kind::Float, image.device())).unsqueeze(0).repeat(&[1, LEVELS as i64]);
            let imj = (i - j).square();
            let idm = glcm / (imj + 1.0);
            f32::from(idm.sum(Kind::Float))
        };

        let difference_variance = {
            let k = Tensor::arange(LEVELS as i64, (Kind::Float, image.device()));
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