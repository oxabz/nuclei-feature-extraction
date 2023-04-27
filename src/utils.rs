use std::marker::PhantomData;
use std::sync::mpsc::{Sender, Receiver, channel};

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

pub struct ParBridgeBridgeS<Iter: Iterator>{
    snd: Sender<Iter::Item>,
    iter: Iter,
}



pub struct ParBridgeBridgeR<Iter: Iterator> {
    rcv: Receiver<Iter::Item>,
}

impl<Iter: Iterator> Iterator for ParBridgeBridgeR<Iter> {
    type Item = Iter::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.rcv.recv().ok()
    }
}

impl<Iter: Iterator> ParBridgeBridgeS<Iter>{
    pub fn run(&mut self) {
        for item in self.iter.by_ref() {
            self.snd.send(item).unwrap();
        }
    }
}

pub trait IntoParBridgeBridge<Iter: Iterator> {
    fn into_par_bridge_bridge(self) -> (ParBridgeBridgeS<Iter>, ParBridgeBridgeR<Iter>);
}

impl<Iter:Iterator> IntoParBridgeBridge<Iter> for Iter{
    fn into_par_bridge_bridge(self) -> (ParBridgeBridgeS<Iter>, ParBridgeBridgeR<Iter>) {
        let (snd, rcv) = channel();
        let bridge = ParBridgeBridgeS {
            snd,
            iter: self,
        };
        let rcv = ParBridgeBridgeR {
            rcv,
        };
        (bridge, rcv)
    }
}