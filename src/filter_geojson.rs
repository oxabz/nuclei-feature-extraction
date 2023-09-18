use std::path::PathBuf;
mod geojson;
use clap::Parser;

#[derive(Debug, Clone, Parser)]
/// Filter element of a geojson file based on an offset and a size
struct Args {
    input: PathBuf,
    output: PathBuf,
    offset_x: usize,
    offset_y: usize,
    size_x: usize,
    size_y: usize,
}

fn main() {
    let Args {
        input,
        output,
        offset_x,
        offset_y,
        size_x,
        size_y,
    } = Args::parse();

    let geometry = std::fs::read_to_string(input).unwrap();
    let mut geometry: geojson::FeatureCollection = serde_json::from_str(&geometry).unwrap();

    geometry.features = geometry
        .features
        .into_iter()
        .filter_map(|feature| {
            let &[left, top, right, down] = feature.bbox.as_slice() else {
                println!("No bbox for {:?}", feature);
                return None;
            };

            if top < offset_y as f32
                || left < offset_x as f32
                || down > (offset_y + size_y) as f32
                || right > (offset_x + size_x) as f32
            {
                return None;
            }

            let mut feature = feature;
            feature.bbox = vec![
                top - offset_y as f32,
                left - offset_x as f32,
                down - offset_y as f32,
                right - offset_x as f32,
            ];

            feature.geometry.coordinates[0]
                .iter_mut()
                .for_each(|point| {
                    point[0] -= offset_x as f32;
                    point[1] -= offset_y as f32;
                });

            Some(feature)
        })
        .collect();

    let geometry = serde_json::to_string(&geometry).unwrap();
    std::fs::write(output, geometry).unwrap();
}
