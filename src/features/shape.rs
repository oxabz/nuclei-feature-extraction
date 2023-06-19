/*!
Contains the shape features computation functions.
 */

use log::{error};
use polars::prelude::*;
use tch::{index::*, Tensor, Kind};

use crate::utils::{PointsExt, centroids_to_key_strings};

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
        assert!(patchs.size().len() == 4, "The patchs tensor must be 4 dimensional");
        assert!(masks.size().len() == 4, "The masks tensor must be 4 dimensional");
        assert!(patchs.size()[1] == 3, "The patchs tensor must have 3 channels");
        assert!(masks.size()[1] == 1, "The masks tensor must have 1 channel");
        assert!(patchs.size()[0] == masks.size()[0], "The number of patchs and masks must be the same");
        assert!(patchs.size()[0] as usize == centroids.len(), "The number of patchs and centroids must be the same");
        assert!(patchs.size()[0] as usize == polygons.len(), "The number of patchs and polygons must be the same");
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

        #[cfg(debug_assertions)]
        let dbg = {
            let h = (batch_size as f64 / 2.0).ceil() as i64;
            Tensor::zeros(&[3, h*64, 4*64], (Kind::Float, device))
        };

        for i in 0..batch_size {
            let polygon = &polygons[i as usize];
            let mask = masks.i(i);

            let hull_mask = tch_utils::shapes::convex_hull(
                patch_size,
                patch_size,
                &polygon.to_tchutils_points(),
                (Kind::Float, device),
            );

            let hull = graham_scan(polygon);

            let mut centroid = center_of_mass(&mask);
            centroid[0] -= patch_size as f32 / 2.0;
            centroid[1] -= patch_size as f32 / 2.0;
            let centroid = centroid;

            #[allow(unused_variables)]
            let (major_axis, minor_axis, angle, eigen) = major_minor_axes_w_angle(&mask);
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

            let convex_perimeter = perimeter(&hull);

            let (convex_hull_area, convex_deffect) =
                convex_hull_stats(&mask, &hull_mask);
            let area_ = area(&mask);
            let perimeter = perimeter(polygon);

            #[cfg(debug_assertions)]
            {
                let x = (i % 2)*128;
                let y = (i / 2)*64;
                let xb = x + 64;
                let patch1 = dbg.i((.., y..y+64, x..x+64));
                let patch2 = dbg.i((.., y..y+64, xb..xb+64));
                patch1.i(0).copy_(&(mask.squeeze()*0.5));
                patch1.i(1).copy_(&(hull_mask.squeeze()*0.5));
                patch2.i(0).copy_(&(mask.squeeze()*0.5));
                patch2.i(1).copy_(&(elipse_mask.squeeze()*0.5));
                let col = Tensor::of_slice(&[1.0, 0.0, 0.0]);
                let ev1 = eigen.i(0);
                // let ev1 = &ev1 / ev1.pow_tensor_scalar(2).sum(tch::Kind::Float).sqrt();
                for l in 0..100{
                    let l = l as f64 / 100.0;
                    let x = (l * f64::from(ev1.i(0)) * major_axis as f64 + centroid[1] as f64 + 31.0).min(63.0) as i64;
                    let y = (l * f64::from(ev1.i(1)) * major_axis as f64 + centroid[0] as f64 + 31.0).min(63.0) as i64;
                    patch2.i((..,y, x)).copy_(&col);
                }
                let col = Tensor::of_slice(&[0.0, 0.0, 1.0]);
                let ev2 = eigen.i(1);
                // let ev2 = &ev2 / ev2.pow_tensor_scalar(2).sum(tch::Kind::Float).sqrt();
                for l in 0..100{
                    let l = l as f64 / 100.0;
                    let x = (l * f64::from(ev2.i(0)) * major_axis as f64 + centroid[1] as f64 + 31.0).min(63.0) as i64;
                    let y = (l * f64::from(ev2.i(1)) * major_axis as f64 + centroid[0] as f64 + 31.0).min(63.0) as i64;
                    patch2.i((..,y, x)).copy_(&col);
                }
                let _ = patch1.i((.., centroid[0] as i64 + 31, centroid[1] as i64 + 31)).fill_(1.0);
                let _ = patch1.i((.., 0, 0)).fill_(1.0);
                let _ = patch1.i((.., 0, 63)).fill_(1.0);
                let _ = patch1.i((.., 63, 0)).fill_(1.0);
                let _ = patch1.i((.., 63, 63)).fill_(1.0);
                let _ = patch2.i((.., centroid[0] as i64 + 31, centroid[1] as i64 + 31)).fill_(1.0);
                let _ = patch2.i((.., 0, 0)).fill_(1.0);
                let _ = patch2.i((.., 0, 63)).fill_(1.0);
                let _ = patch2.i((.., 63, 0)).fill_(1.0);
                let _ = patch2.i((.., 63, 63)).fill_(1.0);
            }

            larea_.push(area_);
            lmajor_axis.push(major_axis);
            lminor_axis.push(minor_axis);
            leccentricity_.push(eccentricity);
            lorientation.push(orientation);
            lperimeter_.push(perimeter);
            lequivalent_perimeter_.push(equivalent_perimeter(area_));
            lcompacity_.push(compacity(area_, perimeter));
            leliptic_deviation_.push(eliptic_deviation(&mask, &elipse_mask));
            lconvex_hull_area.push(convex_hull_area);
            lconvex_deffect.push(convex_deffect);
            lconvex_perimeter.push(convex_perimeter);
        }

        #[cfg(debug_assertions)]
        let _ = tch::vision::image::save(&(dbg*255), "dbg.png");

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
        ).expect("Couldnt create the dataframe")
    }

    fn name(&self)->&str {
        "geometry"
    }
}

pub(crate) fn area(mask: &Tensor) -> f32 {
    f32::from(mask.sum(tch::Kind::Float))
}

pub(crate) fn perimeter(polygon: &Vec<[f32; 2]>) -> f32 {
    let mut perimeter = 0.0;
    for i in 0..polygon.len() {
        let p1 = polygon[i];
        let p2 = polygon[(i + 1) % polygon.len()];
        perimeter += ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2)).sqrt();
    }
    perimeter
}

pub(crate) fn major_minor_axes_w_angle(mask: &Tensor) -> (f32, f32, f32, Tensor) {
    let nz = mask.squeeze().nonzero();
    let centroid = nz.mean_dim(Some(vec![0].as_slice()), false, tch::Kind::Float);

    if centroid.size()[0] == 0 {
        return (f32::NAN, f32::NAN, f32::NAN, Tensor::zeros(&[2, 2], (Kind::Float, mask.device())));
    }

    let points = nz - centroid;
    let cov = points.transpose(0, 1).mm(&points) / points.size()[0];

    if bool::from(cov.isnan().any()) {
        error!("Covariance matrix contains NaN for mask: (dim:{:?}, sum:{})", mask.size(), f32::from(mask.sum(tch::Kind::Float)));
        return (f32::NAN, f32::NAN, f32::NAN, Tensor::zeros(&[2, 2], (Kind::Float, mask.device())));
    }

    if cov.size()[0] != 2 || cov.size()[1] != 2 {
        error!("Covariance matrix is not 2x2 for mask: (dim:{:?}, sum:{})", mask.size(), f32::from(mask.sum(tch::Kind::Float)));
        return (f32::NAN, f32::NAN, f32::NAN, Tensor::zeros(&[2, 2], (Kind::Float, mask.device())));
    }

    if bool::from(cov.isinf().any()) {
        error!("Covariance matrix contains Inf for mask: (dim:{:?}, sum:{})", mask.size(), f32::from(mask.sum(tch::Kind::Float)));
        return (f32::NAN, f32::NAN, f32::NAN, Tensor::zeros(&[2, 2], (Kind::Float, mask.device())));
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

pub(crate) fn convex_hull_stats(mask: &Tensor, hull: &Tensor) -> (f32, f32) {
    let mask = mask.to_kind(tch::Kind::Float);
    let hull = hull.to_kind(tch::Kind::Float);

    let convex_hull_area = area(&hull);

    let convex_deffect = -(area(&mask) - convex_hull_area) / area(&mask);

    (
        convex_hull_area,
        convex_deffect
    )
}

pub(crate) fn graham_scan(points: &[[f32;2]]) -> Vec<[f32;2]> {
    let mut points = points.to_vec();

    let [mut min_x, mut min_y] = points[0];
    let mut min_i = 0;
    for (i, point) in points.iter().enumerate() {
        let [x, y] = *point;
        if y < min_y {
            min_y = y;
            min_x = x;
            min_i = i;
        } else if y == min_y && x < min_x {
            min_x = x;
            min_i = i;
        }
    }
    let p = points.remove(min_i);
    
    let mut points = points
        .into_iter()
        .map(|[x, y]|([x, y], x - p[0], y - p[1]))
        .map(|(p, dx, dy)| (p, (dx.powi(2) + dy.powi(2)).sqrt(), dx))
        .map(|(p, len, dot)| (p, len, - dot / len))
        .filter(|(_, len, _)| *len > 0.0)
        .collect::<Vec<_>>();

    points.sort_by(|(_, al, aa), (_, bl, ba)| aa.partial_cmp(ba).unwrap().then(al.partial_cmp(bl).unwrap()));

    let mut hull = vec![p];
    for (p, _, _) in points {
        while hull.len() > 1 {
            let [ax, ay] = hull[hull.len() - 2];
            let [bx, by] = hull[hull.len() - 1];
            let [cx, cy] = p;
            let ab = (bx - ax, by - ay);
            let bc = (cx - bx, cy - by);
            let cross = ab.0 * bc.1 - ab.1 * bc.0;
            if cross > 0.0 {
                break;
            }
            hull.pop();
        }
        hull.push(p);
    }

    hull 
    
}

pub(crate) fn equivalent_perimeter(area: f32) -> f32 {
    ( area / std::f32::consts::PI ).sqrt() * 2.0 * std::f32::consts::PI
}

pub(crate) fn compacity(area: f32, perimeter: f32) -> f32 {
    area / (perimeter * perimeter) * 4.0 * std::f32::consts::PI
}

pub(crate) fn center_of_mass(mask: &Tensor) -> [f32; 2] {
    let nz = mask.nonzero().i((.., -2..=-1));
    let centroid = nz.mean_dim(Some(vec![0].as_slice()), false, tch::Kind::Float);
    [
        f64::from(centroid.i(vec![0])) as f32,
        f64::from(centroid.i(vec![1])) as f32,
    ]
}
