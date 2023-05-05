mod shape;
mod color;
pub mod texture;

use std::io;

use struct_field_names_as_array::FieldNamesAsArray;
use tch::{Kind, Tensor, index::*};

use crate::{utils::PointsExt, features::color::{circular_mean, circular_std}};

use self::{shape::{center_of_mass, major_minor_axes_w_angle, eccentricity, convex_hull, perimeter, area, equivalent_perimeter, compacity, eliptic_deviation, convex_hull_stats}, color::mean_std, texture::{GlcmFeatures, glcm_features}};

/*
 * Added metrics to Medhi's features:
 * - convex deffect
 * - convex positive deffect
 * - convex negative deffect
 * - perimeter
 * - convex perimeter
 */
#[derive(Debug, serde::Serialize, FieldNamesAsArray)]
pub(crate) struct ShapeFeatures {
    pub(crate) centroid_x: f32,
    pub(crate) centroid_y: f32,
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

impl ShapeFeatures{
    pub fn write_header_to_csv<W: io::Write>(writer: &mut csv::Writer<W>) -> Result<(), csv::Error> {
        writer.write_record(Self::FIELD_NAMES_AS_ARRAY)?;
        Ok(())
    }
}

#[derive(Debug, serde::Serialize, FieldNamesAsArray)]
pub struct ColorFeatures {
    pub centroid_x: f32,
    pub centroid_y: f32,
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
    pub(crate) mean_haematoxylin: f32,
    pub(crate) mean_eosin: f32,
    pub(crate) mean_dab: f32,
    pub(crate) std_haematoxylin : f32,
    pub(crate) std_eosin: f32,
    pub(crate) std_dab: f32, 
}
impl ColorFeatures {
    pub(crate) fn set_all_nan(& mut self) {
        self.mean_r = f32::NAN;
        self.mean_g = f32::NAN;
        self.mean_b = f32::NAN;
        self.std_r = f32::NAN;
        self.std_g = f32::NAN;
        self.std_b = f32::NAN;
        self.mean_h = f32::NAN;
        self.mean_s = f32::NAN;
        self.mean_v = f32::NAN;
        self.std_h = f32::NAN;
        self.std_s = f32::NAN;
        self.std_v = f32::NAN;
        self.mean_haematoxylin = f32::NAN;
        self.mean_eosin = f32::NAN;
        self.mean_dab = f32::NAN;
        self.std_haematoxylin = f32::NAN;
        self.std_eosin = f32::NAN;
        self.std_dab = f32::NAN;
    }

    pub fn write_header_to_csv<W: io::Write>(writer: &mut csv::Writer<W>) -> Result<(), csv::Error> {
        writer.write_record(Self::FIELD_NAMES_AS_ARRAY)?;
        Ok(())
    }
}

pub(crate) fn shape_features(polygon: &Vec<[f32; 2]>, mask: &Tensor) -> ShapeFeatures {
    let device = mask.device();
    let patch_size = mask.size()[2] as usize;
    let hull_mask =
        tch_utils::shapes::convex_hull(patch_size, patch_size, &polygon.to_tchutils_points(), (Kind::Float, device));
    let hull = convex_hull(polygon);

    let mut centroid = center_of_mass(&mask);
    centroid[0] -= patch_size as f32 / 2.0;
    centroid[1] -= patch_size as f32 / 2.0;
    let centroid = centroid;

    let (major_axis, minor_axis, angle) = major_minor_axes_w_angle(&mask);
    let eccentricity = eccentricity(major_axis, minor_axis);
    let orientation = angle;

    let elipse_mask = tch_utils::shapes::ellipse(
        patch_size,
        patch_size,
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
        convex_perimeter,
        centroid_x: 0.0,
        centroid_y: 0.0,
    }
}

pub fn color_features(patch:&Tensor, mask: &Tensor) -> Vec<ColorFeatures>{
    let _ = tch::no_grad_guard();
    let hsv = tch_utils::color::hsv_from_rgb(patch);
    let hed = tch_utils::color::hed_from_rgb(patch);
    let (mean_rgb, std_rgb) = mean_std(patch, mask);
    let (mean_hsv, std_hsv) = mean_std(&hsv, mask);
    let (mean_hed, std_hed) = mean_std(&hed, mask);

    let h = hsv.select(-3, 0);
    let mean_h = circular_mean(&h, &mask);
    let std_h = circular_std(&h, &mask, &mean_h);
    
    let batch_size = patch.size()[0];
    let mut features = Vec::with_capacity(batch_size as usize);
    for i in 0..batch_size {
        let [mean_r, mean_g, mean_b] = Vec::<f32>::from(mean_rgb.i(i))[..] else {
            unreachable!("mean_rgb should be a 2D tensor of size [N, 3]");
        };
        let [std_r, std_g, std_b] = Vec::<f32>::from(std_rgb.i(i))[..] else {
            unreachable!("std_rgb should be a 2D tensor of size [N, 3]");
        };
        let [mean_s, mean_v] = Vec::<f32>::from(mean_hsv.i(i))[1..] else {
            unreachable!("mean_hsv should be a 2D tensor of size [N, 3]");
        };
        let [std_s, std_v] = Vec::<f32>::from(std_hsv.i(i))[1..] else {
            unreachable!("std_hsv should be a 2D tensor of size [N, 3]");
        };
        let [mean_haematoxylin, mean_eosin, mean_dab] = Vec::<f32>::from(mean_hed.i(i))[..] else {
            unreachable!("mean_hed should be a 2D tensor of size [N, 3]");
        };
        let [std_haematoxylin, std_eosin, std_dab] = Vec::<f32>::from(std_hed.i(i))[..] else {
            unreachable!("std_hed should be a 2D tensor of size [N, 3]");
        };
        let mean_h = f32::from(mean_h.i(0));
        let std_h = f32::from(std_h.i(0));

        features.push(ColorFeatures {
            centroid_x:0.0,
            centroid_y:0.0,
            mean_r,
            mean_g,
            mean_b,
            std_r,
            std_g,
            std_b,
            mean_s,
            mean_v,
            std_s,
            std_v,
            mean_haematoxylin,
            mean_eosin,
            mean_dab,
            std_haematoxylin,
            std_eosin,
            std_dab,
            mean_h,
            std_h,
        });
    }
    features
}
