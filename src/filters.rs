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
        output
            .axis_iter_mut(Axis(2))
            .into_par_iter()
            .enumerate()
            .for_each(|(z, mut slice_z)| {
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
        output
            .axis_iter_mut(Axis(2))
            .into_par_iter()
            .enumerate()
            .for_each(|(z, mut slice_z)| {
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
        output
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(x, mut slice_x)| {
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

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    #[test]
    fn gaussian_blur_all_zeros_stays_zero() {
        let arr = Array3::from_elem((5, 5, 5), 0.0f32);
        let out = gaussian_blur_3d(&arr, 1.0);
        assert!(out.iter().all(|&v| v.abs() < EPS));
    }

    #[test]
    fn gaussian_blur_constant_array_unchanged() {
        // Normalized kernel: weighted average of constant C = C.
        let c = 7.0f32;
        let arr = Array3::from_elem((9, 9, 9), c);
        let out = gaussian_blur_3d(&arr, 1.5);
        for &v in out.iter() {
            assert!((v - c).abs() < EPS, "expected {c}, got {v}");
        }
    }

    #[test]
    fn gaussian_blur_single_impulse_peak_at_center() {
        let n = 11usize;
        let mid = n / 2;
        let mut arr = Array3::from_elem((n, n, n), 0.0f32);
        arr[[mid, mid, mid]] = 1.0;
        let out = gaussian_blur_3d(&arr, 1.0);
        let max_val = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!((max_val - out[[mid, mid, mid]]).abs() < EPS);
        assert!(out[[mid + 1, mid, mid]] > EPS);
    }

    #[test]
    fn gaussian_blur_larger_sigma_spreads_more() {
        let n = 15usize;
        let mid = n / 2;
        let mut arr = Array3::from_elem((n, n, n), 0.0f32);
        arr[[mid, mid, mid]] = 1.0;
        let small = gaussian_blur_3d(&arr, 0.5);
        let large = gaussian_blur_3d(&arr, 2.0);
        assert!(small[[mid, mid, mid]] > large[[mid, mid, mid]]);
    }

    #[test]
    fn gaussian_blur_sum_approximately_preserved() {
        // Uniform input with clamp padding: sum is exactly preserved.
        let n = 15usize;
        let arr = Array3::from_elem((n, n, n), 1.0f32);
        let out = gaussian_blur_3d(&arr, 1.0);
        let in_sum: f32 = arr.iter().sum();
        let out_sum: f32 = out.iter().sum();
        let rel_err = ((out_sum - in_sum) / in_sum).abs();
        assert!(rel_err < 0.01, "relative sum error {rel_err} exceeds 1%");
    }
}
