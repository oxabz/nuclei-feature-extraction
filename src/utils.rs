use crate::geojson;

pub type CratePoint = [f32;2];
pub type TchUtilsPoint = (f64, f64);
pub type Points = Vec<CratePoint>;

pub trait CratePointExt {
    fn to_tchutils_point(&self) -> TchUtilsPoint;
    fn into_tchutils_point(self) -> TchUtilsPoint;
}

impl CratePointExt for CratePoint {
    fn to_tchutils_point(&self) -> TchUtilsPoint {
        (self[0] as f64, self[1] as f64)
    }

    fn into_tchutils_point(self) -> TchUtilsPoint {
        (self[0] as f64, self[1] as f64)
    }
}

pub trait PointsExt {
    fn to_tchutils_points(&self) -> Vec<TchUtilsPoint>;
    fn into_tchutils_points(self) -> Vec<TchUtilsPoint>;
}

impl PointsExt for Points {
    fn to_tchutils_points(&self) -> Vec<TchUtilsPoint> {
        self.iter().map(|point| point.to_tchutils_point()).collect()
    }

    fn into_tchutils_points(self) -> Vec<TchUtilsPoint> {
        self.into_iter().map(|point| point.into_tchutils_point()).collect()
    }
}


/**
Preprocess the geojson polygon to extract the centroid and the centered points of the polygon 
 */
pub(crate) fn preprocess_polygon(feature: &geojson::Feature) -> ([f32;2], Vec<[f32; 2]>){
    let polygone = &feature.geometry.coordinates[0];
    let centroid = polygone.iter().fold([0.0,0.0], |mut acc, point| {
        acc[0] += point[0];
        acc[1] += point[1];
        acc
    });
    let centroid = [centroid[0] / polygone.len() as f32, centroid[1] / polygone.len() as f32];
    let centered_polygone = polygone.iter().map(|point|{
        let x = point[0] - centroid[0];
        let y = point[1] - centroid[1];
        [x,y]
    }).collect::<Vec<_>>();
    (centroid, centered_polygone)
}

