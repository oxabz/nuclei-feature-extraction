use tch::{Tensor, index::*};

/**
Compute the mean and standard deviation of the color channels of the image.
# Arguments
- `img` - [N, 3, H, W] tensor
- `mask` - [N, 1, H, W] tensor
# Returns
- `mean` - [N, 3] tensor
 */
pub fn mean_std(img: &Tensor, mask: &Tensor) ->  (Tensor, Tensor){
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
- `mean` - [N, 1] tensor
 */
pub fn circular_mean(image: &Tensor, mask: &Tensor) -> Tensor {
    let mask_area = mask.sum_dim_intlist(Some(&[-1,-2][..]), false, tch::Kind::Float);
    
    let img = image.deg2rad();
    let cos = img.cos() * mask;
    let sin = img.sin() * mask;

    let cos = cos.sum_dim_intlist(Some(&[-1,-2][..]), false, tch::Kind::Float) / &mask_area;
    let sin = sin.sum_dim_intlist(Some(&[-1,-2][..]), false, tch::Kind::Float) / &mask_area;
    
    sin.atan2(&cos).rad2deg()
}

/**
Compute the standard deviation of the image.
# Arguments
- `img` - [N, 1, H, W] tensor
- `mask` - [N, 1, H, W] tensor
- `mean` - [N, 1] tensor
# Returns
- `std` - [N, 1] tensor
 */
pub fn circular_std(image: &Tensor, mask: &Tensor, mean: &Tensor)-> Tensor{
    let mean = mean.view([-1, 1, 1, 1]);
    let centered_img = (image - &mean) * mask;
    let centered_img = centered_img.square();
    let mask_area = mask.sum_dim_intlist(Some(&[-1,-2][..]), false, tch::Kind::Float);
    let variance = centered_img.sum_dim_intlist(Some(&[-1,-2][..]), false, tch::Kind::Float) / &mask_area;
    variance.sqrt()
}