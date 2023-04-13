
use tch::{Tensor, index::*};

use crate::{consts::PATCH_SIZE, mask};

#[derive(Debug)]
pub(crate) struct Features{
    pub(crate) area: f32,
    pub(crate) major_axis: f32,
    pub(crate) minor_axis: f32,
    pub(crate) eccentricity: f32,
    pub(crate) orientation: f32,
}

pub(crate) fn all_features(polygon: &Vec<[f32;2]>, mask:&Tensor)->Features{
    let (major_axis, minor_axis, orientation) = major_minor_axes_w_angle(mask);
    let eccentricity = ((major_axis/2.0).powi(2) - (minor_axis/2.0).powi(2)).sqrt() / (major_axis/2.0);
    
    let convex_hull = mask::poly2mask_convex((PATCH_SIZE, PATCH_SIZE) , polygon);

    Features{
        area: area(mask),
        major_axis,
        minor_axis,
        eccentricity,
        orientation,
    }
}

fn area(mask:&Tensor)->f32{
    f32::from(mask.sum(tch::Kind::Float))
}

fn major_minor_axes_w_angle(mask:&Tensor)->(f32, f32, f32){
    let nz = mask.nonzero().i((.. ,-2..=-1));
    let centroid = nz.mean_dim(Some(vec![0].as_slice()), false, tch::Kind::Float);

    let points = nz - centroid;
    let cov = points.transpose(0, 1).mm(&points) / points.size()[0];

    let (eigenvalues, eigenvector) = cov.linalg_eig();
    
    
    let a = f64::from(eigenvalues.i(vec![0])) as f32;
    let b = f64::from(eigenvalues.i(vec![1])) as f32;

    let (major_axis, minor_axis, mc) = if a > b{
        (a.sqrt(), b.sqrt(), eigenvector.i(0))
    }else{
        (b.sqrt(), a.sqrt(), eigenvector.i(1))
    };

    let x = f64::from(mc.i(vec![0])) as f32;
    let y = f64::from(mc.i(vec![1])) as f32;

    let angle = y.atan2(x);

    (major_axis, minor_axis, angle)
}

