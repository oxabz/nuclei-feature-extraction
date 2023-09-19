/*!
Contains the shape features computation functions.
 */

use log::error;
use polars::prelude::*;
use tch::{index::*, Kind, Tensor};

use crate::utils::{centroids_to_key_strings, PointsExt};

use super::FeatureSet;

pub struct ShapeFeatureSet;

impl FeatureSet for ShapeFeatureSet {
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
        let device = masks.device();
        let batch_size = masks.size()[0];
        let patch_size = masks.size()[3] as usize;

        let mut larea_ = Vec::with_capacity(batch_size as usize);
        let mut lmajor_axis = Vec::with_capacity(batch_size as usize);
        let mut lminor_axis = Vec::with_capacity(batch_size as usize);
        let mut leccentricity_ = Vec::with_capacity(batch_size as usize);
        let mut lorientation = Vec::with_capacity(batch_size as usize);
        let mut lperimeter_ = Vec::with_capacity(batch_size as usize);
        let mut lequivalent_perimeter_ = Vec::with_capacity(batch_size as usize);
        let mut lcompacity_ = Vec::with_capacity(batch_size as usize);
        let mut leliptic_deviation_ = Vec::with_capacity(batch_size as usize);
        let mut lconvex_hull_area = Vec::with_capacity(batch_size as usize);
        let mut lconvex_deffect = Vec::with_capacity(batch_size as usize);
        let mut lconvex_perimeter = Vec::with_capacity(batch_size as usize);

        for i in 0..batch_size {
            let polygon = &polygons[i as usize];
            let polygon = polygon.to_tchutils_points();
            let mask = masks.i(i);

            let mut centroid = center_of_mass(&mask);
            centroid[0] -= patch_size as f32 / 2.0;
            centroid[1] -= patch_size as f32 / 2.0;
            let centroid = centroid;

            let (major_axis, minor_axis, angle, _eigen) = major_minor_axes_w_angle(&mask);
            let eccentricity = eccentricity(major_axis, minor_axis);
            let orientation = angle;

            let elipse_mask = tch_utils::shapes::ellipse(
                patch_size,
                patch_size,
                (centroid[1] as f64, centroid[0] as f64),
                (major_axis as f64, minor_axis as f64),
                angle as f64,
                (Kind::Float, device),
            );

            let area = geometric_features::area(&polygon);
            let perimeter = geometric_features::perimeter(&polygon);
            let equivalent_perimeter = geometric_features::equivalent_perimeter(&polygon);
            let compacity = geometric_features::compacity(&polygon);
            let geometric_features::convex_hull::ConvexHullFeatures {
                area: convex_area,
                perimeter: convex_perimeter,
                deviation: convex_deffect,
            } = geometric_features::convex_hull::convex_hull_features(&polygon);

            larea_.push(area);
            lmajor_axis.push(major_axis);
            lminor_axis.push(minor_axis);
            leccentricity_.push(eccentricity);
            lorientation.push(orientation);
            lperimeter_.push(perimeter);
            lequivalent_perimeter_.push(equivalent_perimeter);
            lcompacity_.push(compacity);
            leliptic_deviation_.push(eliptic_deviation(&mask, &elipse_mask));
            lconvex_hull_area.push(convex_area);
            lconvex_deffect.push(convex_deffect);
            lconvex_perimeter.push(convex_perimeter);
        }

        let centroids = centroids_to_key_strings(centroids);
        df!(
            "centroid" => centroids,
            "area" => larea_,
            "major_axis" => lmajor_axis,
            "minor_axis" => lminor_axis,
            "eccentricity" => leccentricity_,
            "orientation" => lorientation,
            "perimeter" => lperimeter_,
            "equivalent_perimeter" => lequivalent_perimeter_,
            "compacity" => lcompacity_,
            "eliptic_deviation" => leliptic_deviation_,
            "convex_hull_area" => lconvex_hull_area,
            "convex_deffect" => lconvex_deffect,
            "convex_perimeter" => lconvex_perimeter,
        )
        .expect("Couldnt create the dataframe")
    }

    fn name(&self) -> &str {
        "geometry"
    }
}

pub(crate) fn area(mask: &Tensor) -> f32 {
    f32::from(mask.sum(tch::Kind::Float))
}

pub(crate) fn major_minor_axes_w_angle(mask: &Tensor) -> (f32, f32, f32, Tensor) {
    let bail_return: (f32, f32, f32, Tensor) = (
        f32::NAN,
        f32::NAN,
        f32::NAN,
        Tensor::zeros(&[2, 2], (Kind::Float, tch::Device::Cpu)),
    );

    let nz = mask.squeeze().nonzero();
    let centroid = nz.mean_dim(Some(vec![0].as_slice()), false, tch::Kind::Float);

    if centroid.size()[0] == 0 {
        return bail_return;
    }

    let points = nz - centroid;
    let cov = points.transpose(0, 1).mm(&points) / points.size()[0];

    if bool::from(cov.isnan().any()) {
        error!(
            "Covariance matrix contains NaN for mask: (dim:{:?}, sum:{})",
            mask.size(),
            f32::from(mask.sum(tch::Kind::Float))
        );
        return bail_return;
    }

    if cov.size()[0] != 2 || cov.size()[1] != 2 {
        error!(
            "Covariance matrix is not 2x2 for mask: (dim:{:?}, sum:{})",
            mask.size(),
            f32::from(mask.sum(tch::Kind::Float))
        );
        return bail_return;
    }

    if bool::from(cov.isinf().any()) {
        error!(
            "Covariance matrix contains Inf for mask: (dim:{:?}, sum:{})",
            mask.size(),
            f32::from(mask.sum(tch::Kind::Float))
        );
        return bail_return;
    }

    let (eigenvalues, eigenvector) = cov.linalg_eig();

    let a = f64::from(eigenvalues.i(vec![0])) as f32;
    let b = f64::from(eigenvalues.i(vec![1])) as f32;

    let (major_axis, minor_axis, mc) = if a > b {
        (a.sqrt(), b.sqrt(), eigenvector.i(0))
    } else {
        (b.sqrt(), a.sqrt(), eigenvector.i(1))
    };

    let x = f64::from(mc.i(vec![0])) as f32;
    let y = f64::from(mc.i(vec![1])) as f32;

    let angle = x.atan2(y);

    (major_axis * 2.0, minor_axis * 2.0, angle, eigenvector)
}

pub(crate) fn eccentricity(major_axis: f32, minor_axis: f32) -> f32 {
    ((major_axis * 0.5).powi(2) - (minor_axis * 0.5).powi(2)).sqrt() / (major_axis * 0.5)
}

pub(crate) fn eliptic_deviation(mask: &Tensor, elipse_mask: &Tensor) -> f32 {
    let mask = mask.to_kind(tch::Kind::Float);
    let elipse_mask = elipse_mask.to_kind(tch::Kind::Float);

    let mask_area = area(&mask);
    let delta = mask - elipse_mask;

    f64::from(delta.abs().sum(tch::Kind::Float) / mask_area as f64) as f32
}

pub(crate) fn center_of_mass(mask: &Tensor) -> [f32; 2] {
    let nz = mask.nonzero().i((.., -2..=-1));
    let centroid = nz.mean_dim(Some(vec![0].as_slice()), false, tch::Kind::Float);
    [
        f64::from(centroid.i(vec![0])) as f32,
        f64::from(centroid.i(vec![1])) as f32,
    ]
}
