use rayon::prelude::*;


/**
 * Convert a polygon to a mask
 * 
 * # Arguments
 * - `size` - The size of the mask
 * - `polygon` - The polygon to convert
 * 
 * # Returns
 * The mask in a vector of u8 with the layout [row1col1, row1col2, row2col1, row2col2, ...]
 */
pub(crate) fn poly2mask(size:(usize,usize), polygon:&Vec<[f32;2]>)->Vec<u8>{
    let (width, height) = size;

    let mut segments = Vec::new();
    for i in 0..polygon.len(){
        // Store the y coordinates of the segment and the ax+b coefficients 
        let (x1, x2) = (polygon[i][0], polygon[(i+1)%polygon.len()][0]);
        let (y1, y2) = (polygon[i][1], polygon[(i+1)%polygon.len()][1]);
        let a = (y2 - y1) / (x2 - x1 + 1e-7);
        let b = y1 - a * x1;
        segments.push((x1, x2, y1, y2, a, b));
    }
    
    (0..height)
        .into_par_iter()
        .map(|y|{
            let y = y as f32 - height as f32 / 2.0;
            let segments = segments
                .iter()
                .filter(|(_, _, y1, y2, a , _)| (y >= *y1 && y <= *y2 || y >= *y2 && y <= *y1))
                .collect::<Vec<_>>();
            let mut intersection = segments.iter()
                .map(|(x1, x2, _, _, a, b)| {
                    if *a < f32::EPSILON && *a > -f32::EPSILON{
                        return (*x1 + *x2)/2.0;
                    }
                    ((y - b) / a) as f32
                })
                .map(|x| x + width as f32 / 2.0)
                .map(|x| x.max(0.0).min(width as f32))
                .collect::<Vec<f32>>();
            intersection.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mut mask = vec![0; width];
            for i in 0..intersection.len()/2{
                let (start, end) = (intersection[2*i] as usize, intersection[2*i+1]as usize);
                for j in start..end{
                    mask[j] = 255;
                }
            }
            mask
        }).flatten().collect::<Vec<_>>()
}


/**
 * Convert a polygon to a mask of the convex hull
 * 
 * # Arguments
 * - `size` - The size of the mask
 * - `polygon` - The polygon to convert
 * 
 * # Returns
 * The mask in a vector of u8 with the layout [row1col1, row1col2, row2col1, row2col2, ...]
 */
pub(crate) fn poly2mask_convex(size:(usize,usize), polygon:&Vec<[f32;2]>)->Vec<u8>{
    let (width, height) = size;

    let mut segments = Vec::new();
    for i in 0..polygon.len(){
        for j in i+1..polygon.len(){
            // Store the y coordinates of the segment and the ax+b coefficients 
            let (x1, x2) = (polygon[i][0], polygon[j][0]);
            let (y1, y2) = (polygon[i][1], polygon[j][1]);
            let a = (y2 - y1) / (x2 - x1 + 1e-6);
            let b = y1 - a * x1;
            segments.push((y1, y2, a, b));
        }
    }

    (0..height)
        .into_par_iter()
        .map(|y|{
            let y = y as f32 - height as f32 / 2.0;
            let segments = segments
                .iter()
                .filter(|(y1, y2, a, _)| (y >= *y1 && y <= *y2 || y >= *y2 && y <= *y1) && (*a > f32::EPSILON || *a < -f32::EPSILON))
                .collect::<Vec<_>>();
            let mut intersection = segments.iter()
                .map(|(_, _, a, b)| {
                    ((y - b) / a) as f32
                
                
                })
                .map(|x| x + width as f32 / 2.0)
                .map(|x| x.max(0.0).min(width as f32))
                .collect::<Vec<f32>>();
            intersection.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let min = intersection.iter().min_by(|a,b| a.partial_cmp(b).unwrap()).map(|x|*x).unwrap_or(0.0);
            let max = intersection.iter().max_by(|a,b| a.partial_cmp(b).unwrap()).map(|x|*x).unwrap_or(0.0);
            let mut mask = vec![0; width];

            for i in min as usize..max as usize{
                mask[i] = 255;
            }

            mask
        }).flatten().collect::<Vec<_>>()
}
