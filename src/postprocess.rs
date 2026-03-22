use ndarray::Array3;
use rayon::prelude::*;
use crate::masking;
use crate::morphology;

/// Removes small holes from the mask by filling boolean components in the inverted mask
/// that are smaller than `min_size`.
pub fn remove_small_holes(mask: &mut Array3<bool>, min_size: usize) {
    let shape = mask.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    
    // Invert mask to find holes
    let inverted = mask.mapv(|b| !b);
    
    let (labels, sizes) = masking::label_components(&inverted);
    
    // Identify labels to fill (holes)
    let mut labels_to_fill = Vec::new();
    for (idx, &size) in sizes.iter().enumerate() {
        if size < min_size {
            labels_to_fill.push(idx + 1); // labels are 1-based
        }
    }
    
    // Fill holes
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let label = labels[[x, y, z]];
                if label > 0 && labels_to_fill.contains(&label) {
                    mask[[x, y, z]] = true;
                }
            }
        }
    }
}

/// Performs one pass of Gaussian-weighted filling for NaN values.
/// Returns the new array and count of filled voxels.
fn gaussian_fill_pass(data: &Array3<f32>, mask: &Array3<bool>, sigma: f32) -> (Array3<f32>, usize) {
    let shape = data.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    let mut output = data.clone();
    
    // Kernel radius: 3*sigma is standard. For sigma=1, radius=3.
    // Python script uses sigma=1.0.
    let radius = (3.0 * sigma).ceil() as isize;
    
    // Precompute kernel weights
    let mut kernel = Vec::new();
    for x in -radius..=radius {
        for y in -radius..=radius {
            for z in -radius..=radius {
                let dist2 = (x*x + y*y + z*z) as f32;
                if dist2 <= (radius as f32 + 0.5).powi(2) {
                    let weight = (-dist2 / (2.0 * sigma * sigma)).exp();
                    kernel.push((x, y, z, weight));
                }
            }
        }
    }
    
    // Identify voxels to fill: NaN AND inside mask
    // We can iterate over all voxels or just find indices.
    // Since we return a new array, we can parallelize output generation.
    // But we only change NaNs.
    
    // Let's use `par_iter` to compute updates for NaNs.
    // To count filled, we can return Option<f32>.
    
    let updates: Vec<((usize, usize, usize), f32)> = (0..nx * ny * nz).into_par_iter().filter_map(|idx| {
        let z = idx % nz;
        let y = (idx / nz) % ny;
        let x = idx / (ny * nz);
        
        if !mask[[x, y, z]] { return None; }
        if !data[[x, y, z]].is_nan() { return None; }
        
        // It is NaN and inside mask. Try to fill.
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        
        for &(dx, dy, dz, w) in &kernel {
            let ix = x as isize + dx;
            let iy = y as isize + dy;
            let iz = z as isize + dz;
            
            if ix >= 0 && ix < nx as isize &&
               iy >= 0 && iy < ny as isize &&
               iz >= 0 && iz < nz as isize {
                   let val = data[[ix as usize, iy as usize, iz as usize]];
                   if !val.is_nan() {
                       weighted_sum += val * w;
                       weight_sum += w;
                   }
            }
        }
        
        if weight_sum > 0.0 {
            Some(((x, y, z), weighted_sum / weight_sum))
        } else {
            None
        }
    }).collect();
    
    let filled_count = updates.len();
    for ((x, y, z), val) in updates {
        output[[x, y, z]] = val;
    }
    
    (output, filled_count)
}

pub fn fill_holes_iterative(data: &mut Array3<f32>, mask: &Array3<bool>) {
    // Sigma 1.0 as in Python script
    let sigma = 1.0;
    loop {
        let (new_data, count) = gaussian_fill_pass(data, mask, sigma);
        if count == 0 {
            break;
        }
        println!("  Filled {} voxels", count);
        *data = new_data;
    }
}

pub struct PostProcessOptions {
    pub t1_low: f32,
    pub t1_high: f32,
    pub erode_radius: usize,
    pub dilate_radius: usize,
}

impl Default for PostProcessOptions {
    fn default() -> Self {
        Self {
            t1_low: 0.0,
            t1_high: 5500.0,
            erode_radius: 13, // 1.3 * 10
            dilate_radius: 10,
        }
    }
}

pub fn clean_t1_map(
    t1_map: &mut Array3<f32>,
    mask: &mut Array3<bool>,
    options: &PostProcessOptions
) {
    let shape = t1_map.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

    println!("Refining mask...");
    // 1. Remove small holes in the initial mask
    // Python: remove_small_holes(mask, 1000)
    remove_small_holes(mask, 1000);
    
    // 2. Dilate
    *mask = morphology::binary_dilation(mask, options.dilate_radius);
    
    // 3. Erode
    *mask = morphology::binary_erosion(mask, options.erode_radius);
    
    println!("Applying mask and filtering outliers...");
    // 4. Apply mask and filter outliers
    let mut outliers_count = 0;
    let mut mask_removed_count = 0;
    
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                if !mask[[x, y, z]] {
                    if !t1_map[[x, y, z]].is_nan() {
                        t1_map[[x, y, z]] = f32::NAN;
                        mask_removed_count += 1;
                    }
                } else {
                    let val = t1_map[[x, y, z]];
                    if val < options.t1_low || val > options.t1_high {
                        t1_map[[x, y, z]] = f32::NAN;
                        outliers_count += 1;
                    }
                }
            }
        }
    }
    println!("  Removed {} voxels outside mask", mask_removed_count);
    println!("  Removed {} outlier voxels", outliers_count);

    // 5. Fill internal missing values
    println!("Filling holes...");
    fill_holes_iterative(t1_map, mask);
}
