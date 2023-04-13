use serde::{Deserialize};

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Geometry{
  #[serde(rename="type")]
  pub(crate) typ:String,
  pub(crate) coordinates: Vec<Vec<Vec<f32>>>
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Feature {
  pub(crate) bbox: Vec<f32>,
  pub(crate) geometry:Geometry,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct FeatureCollection {
  pub(crate) features: Vec<Feature>,
}