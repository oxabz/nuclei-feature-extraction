use tch::{Tensor, index::*};

/**
Compute the mean and standard deviation of the color channels of the image.
# Arguments
* `img` - [3, H, W] tensor
* `mask` - [1, H, W] tensor
 */
pub fn mean_std(img: &Tensor, mask: &Tensor) ->  ((f32, f32, f32), (f32, f32, f32)){
    let masked /* [3, H, W] */ = img * mask;
    let mask_area /* [3, 1, 1] */ = mask.sum_dim_intlist(Some(&[-1i64, -2][..]), true, tch::Kind::Float);
    let mut mean /* [3, 1, 1] */ = masked.sum_dim_intlist( Some(&[-1i64, -2][..]), true, tch::Kind::Float);
    mean /= &mask_area;

    let std /* [3, H, W] */ = img - &mean;
    let std = &std * &std * mask;

    let mut std = std.sum_dim_intlist(Some(&[-1i64, -2][..]), true, tch::Kind::Float);
    std /= &mask_area;
    std.squeeze_();


    mean.squeeze_();

    let mean = (f64::from(mean.i(0)) as f32, f64::from(mean.i(1)) as f32, f64::from(mean.i(2)) as f32);
    let std = (f64::from(std.i(0)) as f32, f64::from(std.i(1)) as f32, f64::from(std.i(2)) as f32);

    (mean, std)
}

