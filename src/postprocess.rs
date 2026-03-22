use ndarray::Array3;
use rayon::prelude::*;

/// One pass of Gaussian-weighted NaN filling for voxels inside the mask.
/// Returns the updated array and how many voxels were filled.
fn gaussian_fill_pass(data: &Array3<f32>, mask: &Array3<bool>, sigma: f32) -> (Array3<f32>, usize) {
    let shape = data.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

    // Spherical Gaussian kernel, radius = 3 * sigma (matches nan_filter_gaussian in Python).
    let radius = (3.0 * sigma).ceil() as isize;
    let mut kernel = Vec::new();
    for dx in -radius..=radius {
        for dy in -radius..=radius {
            for dz in -radius..=radius {
                let d2 = (dx*dx + dy*dy + dz*dz) as f32;
                if d2 <= (radius as f32 + 0.5).powi(2) {
                    kernel.push((dx, dy, dz, (-d2 / (2.0 * sigma * sigma)).exp()));
                }
            }
        }
    }

    let updates: Vec<((usize, usize, usize), f32)> = (0..nx * ny * nz)
        .into_par_iter()
        .filter_map(|idx| {
            let z = idx % nz;
            let y = (idx / nz) % ny;
            let x = idx / (ny * nz);

            if !mask[[x, y, z]] || !data[[x, y, z]].is_nan() {
                return None;
            }

            let mut weighted_sum = 0.0f32;
            let mut weight_sum   = 0.0f32;
            for &(dx, dy, dz, w) in &kernel {
                let ix = x as isize + dx;
                let iy = y as isize + dy;
                let iz = z as isize + dz;
                if ix >= 0 && ix < nx as isize &&
                   iy >= 0 && iy < ny as isize &&
                   iz >= 0 && iz < nz as isize {
                    let v = data[[ix as usize, iy as usize, iz as usize]];
                    if !v.is_nan() {
                        weighted_sum += v * w;
                        weight_sum   += w;
                    }
                }
            }

            if weight_sum > 0.0 {
                Some(((x, y, z), weighted_sum / weight_sum))
            } else {
                None
            }
        })
        .collect();

    let filled = updates.len();
    let mut output = data.clone();
    for ((x, y, z), val) in updates {
        output[[x, y, z]] = val;
    }
    (output, filled)
}

/// Iteratively fills NaN voxels inside the mask using a Gaussian-weighted average of
/// neighbouring finite values (sigma = 1.0, matching nan_filter_gaussian in Python).
pub fn fill_holes_iterative(data: &mut Array3<f32>, mask: &Array3<bool>) {
    loop {
        let (new_data, count) = gaussian_fill_pass(data, mask, 1.0);
        if count == 0 { break; }
        println!("  Filled {} voxels", count);
        *data = new_data;
    }
}

pub struct PostProcessOptions {
    pub t1_low:  f32,
    pub t1_high: f32,
}

impl Default for PostProcessOptions {
    fn default() -> Self {
        Self { t1_low: 0.0, t1_high: 5500.0 }
    }
}

/// Applies the face mask, removes T1 outliers, checks that the result is non-empty,
/// then fills any remaining NaN voxels inside the mask by Gaussian interpolation.
pub fn clean_t1_map(
    t1_map:  &mut Array3<f32>,
    mask:    &Array3<bool>,
    options: &PostProcessOptions,
) {
    let shape = t1_map.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

    println!("Applying mask and filtering outliers...");
    let mut mask_removed = 0usize;
    let mut outliers     = 0usize;

    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                if !mask[[x, y, z]] {
                    if !t1_map[[x, y, z]].is_nan() {
                        t1_map[[x, y, z]] = f32::NAN;
                        mask_removed += 1;
                    }
                } else {
                    let v = t1_map[[x, y, z]];
                    if v < options.t1_low || v > options.t1_high {
                        t1_map[[x, y, z]] = f32::NAN;
                        outliers += 1;
                    }
                }
            }
        }
    }
    println!("  Removed {} voxels outside mask", mask_removed);
    println!("  Removed {} outlier voxels", outliers);

    // Guard against bad units / extreme T1 bounds.
    let total        = nx * ny * nz;
    let finite_count = t1_map.iter().filter(|v| v.is_finite()).count();
    if (finite_count as f32 / total as f32) < 0.01 {
        panic!(
            "After outlier removal, less than 1% of the image is finite \
             ({}/{} voxels). Check units or t1_low/t1_high (currently {}/{}).",
            finite_count, total, options.t1_low, options.t1_high
        );
    }

    println!("Filling holes...");
    fill_holes_iterative(t1_map, mask);
}
