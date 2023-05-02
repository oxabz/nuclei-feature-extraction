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
