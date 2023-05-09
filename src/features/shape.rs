/*!
Contains the shape features computation functions.
 */

use tch::{Tensor, index::*};



pub(crate) fn area(mask: &Tensor) -> f32 {
    f32::from(mask.sum(tch::Kind::Float))
}

pub(crate) fn perimeter(polygon: &Vec<[f32; 2]>) -> f32{
    let mut perimeter = 0.0;
    for i in 0..polygon.len() {
        let p1 = polygon[i];
        let p2 = polygon[(i + 1) % polygon.len()];
        perimeter += ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2)).sqrt();
    }
    perimeter
}

pub(crate) fn major_minor_axes_w_angle(mask: &Tensor) -> (f32, f32, f32) {
    let nz = mask.squeeze().nonzero();
    let centroid = nz.mean_dim(Some(vec![0].as_slice()), false, tch::Kind::Float);

    let points = nz - centroid;
    let cov = points.transpose(0, 1).mm(&points) / points.size()[0];

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

    let angle = y.atan2(x);

    (major_axis, minor_axis, angle)
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

pub(crate) fn convex_hull_stats(mask: &Tensor, hull: &Tensor) -> (f32, f32, f32, f32) {
    let mask = mask.to_kind(tch::Kind::Float);
    let hull = hull.to_kind(tch::Kind::Float);

    let convex_hull_area = area(&hull);

    let delta = &mask - &hull;

    let convex_deffect = (area(&mask) - convex_hull_area) / area(&mask);

    let convex_positive_defect =
        f64::from(delta.clamp_min(0.0).sum(tch::Kind::Float)) as f32 / area(&mask);
    let convex_negative_defect =
        f64::from((-delta).clamp_max(0.0).sum(tch::Kind::Float)) as f32 / area(&mask);

    (
        convex_hull_area,
        convex_deffect,
        convex_positive_defect,
        convex_negative_defect,
    )
}

pub(crate) fn convex_hull(points: &Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    // Creating the convex hull with the graham scan algorithm
    assert!(points.len() > 2, "Convex hull must have at least 3 points");
    let mut points = points.clone();
    let p = points
        .iter().enumerate()
        .min_by(|(_,a), (_, b)| {
            a[1].partial_cmp(&b[1])
                .unwrap()
                .then(a[0].partial_cmp(&b[0]).unwrap())
        })
        .unwrap().0;
    let p = points.remove(p);
    
    points.sort_by(|a, b| {
        // Using cos to sorts points by angle
        let aa = (a[1] - p[1]).cos();
        let ab = (b[1] - p[1]).cos();
        
        aa.partial_cmp(&ab).unwrap().then_with(|| {
            // Using manhattan distance to break ties
            let da = (p[0] - a[0]).abs().max((p[1] - a[1]).abs());
            let db = (p[0] - b[0]).abs().max((p[1] - b[1]).abs());
            db.partial_cmp(&da).unwrap()
        })
    });

    let mut hull = vec![p, points[0]];
    for p in points.iter().skip(1) {
        while hull.len() > 1 {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            let c = *p;
            let cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            if cross < 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(*p);
    }
    hull
}

pub(crate) fn equivalent_perimeter(area:f32) -> f32 {
    (std::f32::consts::PI / area).sqrt() * 2.0
}

pub(crate) fn compacity(area:f32, perimeter:f32) -> f32 {
    area / (perimeter * perimeter) * 4.0 * std::f32::consts::PI
}

pub(crate) fn center_of_mass(mask: &Tensor) -> [f32; 2] {
    let nz = mask.nonzero().i((.., -2..=-1));
    let centroid = nz.mean_dim(Some(vec![0].as_slice()), false, tch::Kind::Float);
    [f64::from(centroid.i(vec![0])) as f32, f64::from(centroid.i(vec![1])) as f32]
}