
use tch::{Tensor, index::*, Kind};

use crate::{consts::PATCH_SIZE, utils::PointsExt};

const DBG: bool = true;

#[derive(Debug)]
pub(crate) struct Features{
    pub(crate) area: f32,
    pub(crate) major_axis: f32,
    pub(crate) minor_axis: f32,
    pub(crate) eccentricity: f32,
    pub(crate) orientation: f32,
}


pub(crate) fn all_features(polygon: &Vec<[f32;2]>, patch:&Tensor)->Features{
    let polygon = polygon.to_tchutils_points();
    let device = patch.device();
    let mask = tch_utils::shapes::polygon(PATCH_SIZE, PATCH_SIZE, &polygon, (Kind::Float, device));
    let hull = tch_utils::shapes::convex_hull(PATCH_SIZE, PATCH_SIZE, &polygon, (Kind::Float, device));

    let nz = mask.nonzero().i((.. ,-2..=-1));
    let centroid = nz.mean_dim(Some(vec![0].as_slice()), false, tch::Kind::Float);
    let centroid = centroid - (0.5 * PATCH_SIZE as f64);
    let centroid = (
        f64::from(centroid.i(0)),
        f64::from(centroid.i(1)),
    );

    let (major_axis, minor_axis, angle) = major_minor_axes_w_angle(&mask);
    let eccentricity = eccentricity(major_axis, minor_axis);
    let orientation = angle;

    let elipse_mask = tch_utils::shapes::ellipse(PATCH_SIZE, PATCH_SIZE, centroid, (major_axis as f64 * 2.0, minor_axis as f64 * 2.0), angle as f64, (Kind::Float, device));

    if DBG {
        let c = Tensor::cat(&[&mask, &hull, &elipse_mask], 2).repeat(&[3, 1, 1]);
        // display points
        for point in polygon{
            let x = (point.0 as i64 + (PATCH_SIZE as i64 / 2)).clamp(0, PATCH_SIZE as i64 - 1);
            let y = (point.1 as i64 + (PATCH_SIZE as i64 / 2)).clamp(0, PATCH_SIZE as i64 - 1);

            c.i((0, y, x)).fill_(1.0);
            c.i((0, y, x + PATCH_SIZE as i64)).fill_(1.0);
            c.i((0, y, x + (PATCH_SIZE*2) as i64)).fill_(1.0);

            c.i((1, y, x)).fill_(0.0);
            c.i((1, y, x + PATCH_SIZE as i64)).fill_(0.0);
            c.i((1, y, x + (PATCH_SIZE*2) as i64)).fill_(0.0);

            c.i((2, y, x)).fill_(0.0);
            c.i((2, y, x + PATCH_SIZE as i64)).fill_(0.0);
            c.i((2, y, x + (PATCH_SIZE*2) as i64)).fill_(0.0);
        }

        let dbg = tch::vision::image::load("dbg.png").ok();

        match dbg {
            Some(dbg) if dbg.size()[1] < (PATCH_SIZE * 25) as i64 => {
                let dbg = Tensor::cat(&[dbg, c*255], 1);
                tch::vision::image::save(&dbg, "dbg.png").unwrap();
            },
            None => {
                tch::vision::image::save(&c, "dbg.png").unwrap();
            }
            _ =>{}
        }
    }

    Features{
        area: area(&mask),
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

fn eccentricity(major_axis:f32, minor_axis:f32)->f32{
    ((major_axis*0.5).powi(2) - (minor_axis*0.5).powi(2)).sqrt() / (major_axis*0.5)
}