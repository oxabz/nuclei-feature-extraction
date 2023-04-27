mod shape;
mod color;
mod texture;

use tch::{Kind, Tensor};

use crate::{consts::PATCH_SIZE, utils::PointsExt};

use self::{shape::{center_of_mass, major_minor_axes_w_angle, eccentricity, convex_hull, perimeter, area, equivalent_perimeter, compacity, eliptic_deviation, convex_hull_stats}, color::mean_std, texture::{GlcmFeatures, glcm_features}};

/*
 * Added metrics to Medhi's features:
 * - convex deffect
 * - convex positive deffect
 * - convex negative deffect
 * - perimeter
 * - convex perimeter
 */
#[derive(Debug, serde::Serialize)]
pub(crate) struct ShapeFeatures {
    pub(crate) area: f32,
    pub(crate) perimeter: f32,
    pub(crate) equivalent_perimeter: f32,
    pub(crate) major_axis: f32,
    pub(crate) minor_axis: f32,
    pub(crate) eccentricity: f32,
    pub(crate) orientation: f32,
    pub(crate) eliptic_deviation: f32,
    pub(crate) convex_hull_area: f32,
    pub(crate) convex_deffect: f32,
    pub(crate) convex_positive_defect: f32,
    pub(crate) convex_negative_defect: f32,
    pub(crate) convex_perimeter: f32,
    pub(crate) compacity: f32,
}

pub struct ColorFeatures {
    // RGB
    pub(crate) mean_r: f32,
    pub(crate) mean_g: f32,
    pub(crate) mean_b: f32,
    pub(crate) std_r: f32,
    pub(crate) std_g: f32,
    pub(crate) std_b: f32,
    // HSV 
    pub(crate) mean_h: f32,
    pub(crate) mean_s: f32,
    pub(crate) mean_v: f32,
    pub(crate) std_h: f32,
    pub(crate) std_s: f32,
    pub(crate) std_v: f32,
    // HED
    pub(crate) mean_hematoxylin: f32,
    pub(crate) mean_eosine: f32,
    pub(crate) mean_dab: f32,
    pub(crate) std_hematoxykin : f32,
    pub(crate) std_eosine: f32,
    pub(crate) std_dab: f32, 
}

/*
Added metrics to Medhi's features:
- shapes
    - convex deffect
    - convex positive deffect
    - convex negative deffect
    - perimeter
    - convex perimeter

 */
#[derive(Debug, serde::Serialize)]
pub(crate) struct Features {
    // Shape features
    pub(crate) area: f32,
    pub(crate) perimeter: f32,
    pub(crate) equivalent_perimeter: f32,
    pub(crate) major_axis: f32,
    pub(crate) minor_axis: f32,
    pub(crate) eccentricity: f32,
    pub(crate) orientation: f32,
    pub(crate) eliptic_deviation: f32,
    pub(crate) convex_hull_area: f32,
    pub(crate) convex_deffect: f32,
    pub(crate) convex_positive_defect: f32,
    pub(crate) convex_negative_defect: f32,
    pub(crate) convex_perimeter: f32,
    pub(crate) compacity: f32,
    // Color features
    pub(crate) mean_r: f32,
    pub(crate) mean_g: f32,
    pub(crate) mean_b: f32,
    pub(crate) std_r: f32,
    pub(crate) std_g: f32,
    pub(crate) std_b: f32,
    pub(crate) mean_h: f32,
    pub(crate) mean_s: f32,
    pub(crate) mean_v: f32,
    pub(crate) std_h: f32,
    pub(crate) std_s: f32,
    pub(crate) std_v: f32,
    pub(crate) mean_hematoxylin: f32,
    pub(crate) mean_eosine: f32,
    pub(crate) mean_dab: f32,
    pub(crate) std_hematoxykin : f32,
    pub(crate) std_eosine: f32,
    pub(crate) std_dab: f32, 
    // GLCM features
    pub glcm0_correlation: f32,
    pub glcm0_contraste: f32,
    pub glcm0_dissimilarity: f32,
    pub glcm0_entropy: f32,
    pub glcm0_angular_second_moment: f32,
    pub glcm0_sum_average: f32,
    pub glcm0_sum_variance: f32,
    pub glcm0_sum_entropy: f32,
    pub glcm0_sum_of_squares: f32,
    pub glcm0_inverse_difference_moment: f32,
    pub glcm0_difference_variance: f32,
    pub glcm0_information_measure_correlation1: f32,
    pub glcm0_information_measure_correlation2: f32,
    pub glcm45_correlation: f32,
    pub glcm45_contraste: f32,
    pub glcm45_dissimilarity: f32,
    pub glcm45_entropy: f32,
    pub glcm45_angular_second_moment: f32,
    pub glcm45_sum_average: f32,
    pub glcm45_sum_variance: f32,
    pub glcm45_sum_entropy: f32,
    pub glcm45_sum_of_squares: f32,
    pub glcm45_inverse_difference_moment: f32,
    pub glcm45_difference_variance: f32,
    pub glcm45_information_measure_correlation1: f32,
    pub glcm45_information_measure_correlation2: f32,
    pub glcm90_correlation: f32,
    pub glcm90_contraste: f32,
    pub glcm90_dissimilarity: f32,
    pub glcm90_entropy: f32,
    pub glcm90_angular_second_moment: f32,
    pub glcm90_sum_average: f32,
    pub glcm90_sum_variance: f32,
    pub glcm90_sum_entropy: f32,
    pub glcm90_sum_of_squares: f32,
    pub glcm90_inverse_difference_moment: f32,
    pub glcm90_difference_variance: f32,
    pub glcm90_information_measure_correlation1: f32,
    pub glcm90_information_measure_correlation2: f32,
    pub glcm135_correlation: f32,
    pub glcm135_contraste: f32,
    pub glcm135_dissimilarity: f32,
    pub glcm135_entropy: f32,
    pub glcm135_angular_second_moment: f32,
    pub glcm135_sum_average: f32,
    pub glcm135_sum_variance: f32,
    pub glcm135_sum_entropy: f32,
    pub glcm135_sum_of_squares: f32,
    pub glcm135_inverse_difference_moment: f32,
    pub glcm135_difference_variance: f32,
    pub glcm135_information_measure_correlation1: f32,
    pub glcm135_information_measure_correlation2: f32,
    
}

pub(crate) fn all_features(polygon: &Vec<[f32; 2]>, patch: &Tensor) -> Features {
    let device = patch.device();
    let mask = tch_utils::shapes::polygon(PATCH_SIZE, PATCH_SIZE, &polygon.to_tchutils_points(), (Kind::Float, device));

    let ShapeFeatures {
        area,
        perimeter,
        equivalent_perimeter,
        major_axis,
        minor_axis,
        eccentricity,
        orientation,
        eliptic_deviation,
        convex_hull_area,
        convex_deffect,
        convex_positive_defect,
        convex_negative_defect,
        convex_perimeter,
        compacity,
    } = shape_features(polygon, &mask);

    let ColorFeatures{
        mean_r,
        mean_g,
        mean_b,
        std_r,
        std_g,
        std_b,
        mean_h,
        mean_s,
        mean_v,
        std_h,
        std_s,
        std_v,
        mean_hematoxylin,
        mean_eosine,
        mean_dab,
        std_hematoxykin,
        std_eosine,
        std_dab,
    } = color_features(patch, &mask);

    let glcms = glcm_features(patch, &mask);

    let GlcmFeatures{
        correlation: glcm0_correlation,
        contraste: glcm0_contraste,
        dissimilarity: glcm0_dissimilarity,
        entropy: glcm0_entropy,
        angular_second_moment: glcm0_angular_second_moment,
        sum_average: glcm0_sum_average,
        sum_variance: glcm0_sum_variance,
        sum_entropy: glcm0_sum_entropy,
        sum_of_squares: glcm0_sum_of_squares,
        inverse_difference_moment: glcm0_inverse_difference_moment,
        difference_variance: glcm0_difference_variance,
        information_measure_correlation1: glcm0_information_measure_correlation1,
        information_measure_correlation2: glcm0_information_measure_correlation2,
    } = glcms[0];

    let GlcmFeatures{
        correlation: glcm45_correlation,
        contraste: glcm45_contraste,
        dissimilarity: glcm45_dissimilarity,
        entropy: glcm45_entropy,
        angular_second_moment: glcm45_angular_second_moment,
        sum_average: glcm45_sum_average,
        sum_variance: glcm45_sum_variance,
        sum_entropy: glcm45_sum_entropy,
        sum_of_squares: glcm45_sum_of_squares,
        inverse_difference_moment: glcm45_inverse_difference_moment,
        difference_variance: glcm45_difference_variance,
        information_measure_correlation1: glcm45_information_measure_correlation1,
        information_measure_correlation2: glcm45_information_measure_correlation2,
    } = glcms[1];

    let GlcmFeatures{
        correlation: glcm90_correlation,
        contraste: glcm90_contraste,
        dissimilarity: glcm90_dissimilarity,
        entropy: glcm90_entropy,
        angular_second_moment: glcm90_angular_second_moment,
        sum_average: glcm90_sum_average,
        sum_variance: glcm90_sum_variance,
        sum_entropy: glcm90_sum_entropy,
        sum_of_squares: glcm90_sum_of_squares,
        inverse_difference_moment: glcm90_inverse_difference_moment,
        difference_variance: glcm90_difference_variance,
        information_measure_correlation1: glcm90_information_measure_correlation1,
        information_measure_correlation2: glcm90_information_measure_correlation2,
    } = glcms[2];

    let GlcmFeatures{
        correlation: glcm135_correlation,
        contraste: glcm135_contraste,
        dissimilarity: glcm135_dissimilarity,
        entropy: glcm135_entropy,
        angular_second_moment: glcm135_angular_second_moment,
        sum_average: glcm135_sum_average,
        sum_variance: glcm135_sum_variance,
        sum_entropy: glcm135_sum_entropy,
        sum_of_squares: glcm135_sum_of_squares,
        inverse_difference_moment: glcm135_inverse_difference_moment,
        difference_variance: glcm135_difference_variance,
        information_measure_correlation1: glcm135_information_measure_correlation1,
        information_measure_correlation2: glcm135_information_measure_correlation2,
    } = glcms[3];

    Features {
        area,
        perimeter,
        equivalent_perimeter,
        major_axis,
        minor_axis,
        eccentricity,
        orientation,
        eliptic_deviation,
        convex_hull_area,
        convex_deffect,
        convex_positive_defect,
        convex_negative_defect,
        convex_perimeter,
        compacity,
        mean_r,
        mean_g,
        mean_b,
        std_r,
        std_g,
        std_b,
        mean_h,
        mean_s,
        mean_v,
        std_h,
        std_s,
        std_v,
        mean_hematoxylin,
        mean_eosine,
        mean_dab,
        std_hematoxykin,
        std_eosine,
        std_dab,
        glcm0_correlation,
        glcm0_contraste,
        glcm0_dissimilarity,
        glcm0_entropy,
        glcm0_angular_second_moment,
        glcm0_sum_average,
        glcm0_sum_variance,
        glcm0_sum_entropy,
        glcm0_sum_of_squares,
        glcm0_inverse_difference_moment,
        glcm0_difference_variance,
        glcm0_information_measure_correlation1,
        glcm0_information_measure_correlation2,
        glcm45_correlation,
        glcm45_contraste,
        glcm45_dissimilarity,
        glcm45_entropy,
        glcm45_angular_second_moment,
        glcm45_sum_average,
        glcm45_sum_variance,
        glcm45_sum_entropy,
        glcm45_sum_of_squares,
        glcm45_inverse_difference_moment,
        glcm45_difference_variance,
        glcm45_information_measure_correlation1,
        glcm45_information_measure_correlation2,
        glcm90_correlation,
        glcm90_contraste,
        glcm90_dissimilarity,
        glcm90_entropy,
        glcm90_angular_second_moment,
        glcm90_sum_average,
        glcm90_sum_variance,
        glcm90_sum_entropy,
        glcm90_sum_of_squares,
        glcm90_inverse_difference_moment,
        glcm90_difference_variance,
        glcm90_information_measure_correlation1,
        glcm90_information_measure_correlation2,
        glcm135_correlation,
        glcm135_contraste,
        glcm135_dissimilarity,
        glcm135_entropy,
        glcm135_angular_second_moment,
        glcm135_sum_average,
        glcm135_sum_variance,
        glcm135_sum_entropy,
        glcm135_sum_of_squares,
        glcm135_inverse_difference_moment,
        glcm135_difference_variance,
        glcm135_information_measure_correlation1,
        glcm135_information_measure_correlation2,
    }
}

pub(crate) fn shape_features(polygon: &Vec<[f32; 2]>, mask: &Tensor) -> ShapeFeatures {
    let device = mask.device();
    let hull_mask =
        tch_utils::shapes::convex_hull(PATCH_SIZE, PATCH_SIZE, &polygon.to_tchutils_points(), (Kind::Float, device));
    let hull = convex_hull(polygon);

    let mut centroid = center_of_mass(&mask);
    centroid[0] -= PATCH_SIZE as f32 / 2.0;
    centroid[1] -= PATCH_SIZE as f32 / 2.0;
    let centroid = centroid;

    let (major_axis, minor_axis, angle) = major_minor_axes_w_angle(&mask);
    let eccentricity = eccentricity(major_axis, minor_axis);
    let orientation = angle;

    let elipse_mask = tch_utils::shapes::ellipse(
        PATCH_SIZE,
        PATCH_SIZE,
        (centroid[0] as f64, centroid[1] as f64),
        (major_axis as f64 * 2.0, minor_axis as f64 * 2.0),
        angle as f64,
        (Kind::Float, device),
    );

    let convex_perimeter = perimeter(&hull);

    let (convex_hull_area, convex_deffect, convex_positive_defect, convex_negative_defect) = convex_hull_stats(&mask, &hull_mask);
    let area_ = area(&mask);
    let perimeter = perimeter(&polygon);
    ShapeFeatures {
        area: area_,
        major_axis,
        minor_axis,
        eccentricity,
        orientation,
        perimeter,
        equivalent_perimeter: equivalent_perimeter(area_),
        compacity: compacity(area_, perimeter),
        eliptic_deviation: eliptic_deviation(&mask, &elipse_mask),
        convex_hull_area,
        convex_deffect,
        convex_positive_defect,
        convex_negative_defect,
        convex_perimeter
    }
}

pub fn color_features(patch:&Tensor, mask: &Tensor) -> ColorFeatures{
    let hsv = tch_utils::color::hsv_from_rgb(patch);
    let hed = tch_utils::color::hed_from_rgb(patch);

    let ((mean_r, mean_g, mean_b), (std_r, std_g, std_b)) = mean_std(patch, mask);
    let ((_, mean_s, mean_v), (std_h, std_s, std_v)) = mean_std(&hsv, mask);
    let ((mean_hematoxylin, mean_eosine, mean_dab), (std_hematoxykin, std_eosine, std_dab)) = mean_std(&hed, mask);

    let h = hsv.select(-3, 0);
    let c = f64::from(h.cos().mean(Kind::Float));
    let s = f64::from(h.sin().mean(Kind::Float));
    let mean_h = c.atan2(s) as f32;

    ColorFeatures {
        mean_r,
        mean_g,
        mean_b,
        std_r,
        std_g,
        std_b,
        mean_h,
        mean_s,
        mean_v,
        std_h,
        std_s,
        std_v,
        mean_hematoxylin,
        mean_eosine,
        mean_dab,
        std_hematoxykin,
        std_eosine,
        std_dab,
    }
}
