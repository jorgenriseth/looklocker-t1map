use ndarray::{Array3, Axis};
use rayon::prelude::*;

fn gaussian_kernel_1d(sigma: f32) -> Vec<f32> {
    // skimage.filters.gaussian defaults to truncate=4.0 (radius = 4 * sigma).
    // Using 3 * sigma gives a narrower kernel that blurs less at the mask boundary
    // and produces a slightly different ISODATA threshold.
    let radius = (4.0 * sigma).ceil() as isize;
    let mut kernel = Vec::new();
    let mut sum = 0.0;
    
    for i in -radius..=radius {
        let x = i as f32;
        let val = (-(x * x) / (2.0 * sigma * sigma)).exp();
        kernel.push(val);
        sum += val;
    }
    
    // Normalize
    for val in &mut kernel {
        *val /= sum;
    }
    
    kernel
}

fn convolve_1d(input: &Array3<f32>, axis: usize, kernel: &[f32]) -> Array3<f32> {
    let shape = input.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    let mut output = Array3::from_elem((nx, ny, nz), 0.0);
    let radius = (kernel.len() / 2) as isize;
    
    if axis == 0 {
        // Convolve along X. Parallelize over Z.
        output.axis_iter_mut(Axis(2)).into_par_iter().enumerate().for_each(|(z, mut slice_z)| {
            for y in 0..ny {
                for x in 0..nx {
                    let mut sum = 0.0;
                    for k in 0..kernel.len() {
                        let offset = k as isize - radius;
                        let ix = x as isize + offset;
                        let ix_clamped = ix.max(0).min(nx as isize - 1) as usize;
                        sum += input[[ix_clamped, y, z]] * kernel[k];
                    }
                    slice_z[[x, y]] = sum;
                }
            }
        });
    } else if axis == 1 {
        // Convolve along Y. Parallelize over Z.
        output.axis_iter_mut(Axis(2)).into_par_iter().enumerate().for_each(|(z, mut slice_z)| {
            for x in 0..nx {
                for y in 0..ny {
                    let mut sum = 0.0;
                    for k in 0..kernel.len() {
                        let offset = k as isize - radius;
                        let iy = y as isize + offset;
                        let iy_clamped = iy.max(0).min(ny as isize - 1) as usize;
                        sum += input[[x, iy_clamped, z]] * kernel[k];
                    }
                    slice_z[[x, y]] = sum;
                }
            }
        });
    } else {
        // Convolve along Z. Parallelize over X.
        output.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(x, mut slice_x)| {
            for y in 0..ny {
                for z in 0..nz {
                    let mut sum = 0.0;
                    for k in 0..kernel.len() {
                        let offset = k as isize - radius;
                        let iz = z as isize + offset;
                        let iz_clamped = iz.max(0).min(nz as isize - 1) as usize;
                        sum += input[[x, y, iz_clamped]] * kernel[k];
                    }
                    slice_x[[y, z]] = sum;
                }
            }
        });
    }
    
    output
}

pub fn gaussian_blur_3d(input: &Array3<f32>, sigma: f32) -> Array3<f32> {
    let kernel = gaussian_kernel_1d(sigma);
    let pass1 = convolve_1d(input, 0, &kernel);
    let pass2 = convolve_1d(&pass1, 1, &kernel);
    convolve_1d(&pass2, 2, &kernel)
}
