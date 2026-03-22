use ndarray::Array3;
use rayon::prelude::*;

/// Generates a spherical structuring element (list of offsets) for a given radius.
fn get_ball_offsets(radius: usize) -> Vec<(isize, isize, isize)> {
    let r = radius as isize;
    let mut offsets = Vec::new();
    for x in -r..=r {
        for y in -r..=r {
            for z in -r..=r {
                if (x*x + y*y + z*z) as f64 <= (r as f64 + 0.5).powi(2) {
                    offsets.push((x, y, z));
                }
            }
        }
    }
    offsets
}

/// 3D Binary Dilation
pub fn binary_dilation(input: &Array3<bool>, radius: usize) -> Array3<bool> {
    let shape = input.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    let offsets = get_ball_offsets(radius);

    // Naive parallel implementation: iterate over output voxels
    // Optimization: Only iterate over true input voxels and paint neighbors?
    // "Paint neighbors" requires atomic or mutex, which is slow.
    // "Check neighbors" for each output voxel is parallel-friendly (read-only input).
    
    // Better: For each voxel in output, if any neighbor in input is true, set true.
    // But this is slow if kernel is large. Radius is usually small.
    
    // Let's stick to "for each voxel, check neighbors".
    // We can iterate over indices.
    
    // Use unsafe raw slice or Zip for performance if needed, but let's start safe.
    // To make it parallel, we can collect results or use `Array3::from_shape_vec`.
    
    let out_vec: Vec<bool> = (0..nx * ny * nz).into_par_iter().map(|idx| {
        // Convert flat index to 3D
        let z = idx % nz;
        let y = (idx / nz) % ny;
        let x = idx / (ny * nz);
        
        // If input is true, output is true (dilation includes center usually).
        // Actually mathematical dilation: if any input pixel in kernel is 1.
        // If center is 1, it stays 1 (assuming kernel contains (0,0,0)).
        
        // Optimization: if input at center is true, we don't need to check neighbors (if kernel has origin).
        // Standard structuring element usually includes origin.
        if input[[x, y, z]] {
            return true;
        }

        // Check neighbors
        for &(dx, dy, dz) in &offsets {
            let ix = x as isize + dx;
            let iy = y as isize + dy;
            let iz = z as isize + dz;
            
            if ix >= 0 && ix < nx as isize &&
               iy >= 0 && iy < ny as isize &&
               iz >= 0 && iz < nz as isize
                   && input[[ix as usize, iy as usize, iz as usize]] {
                       return true;
                   }
        }
        false
    }).collect();
    
    Array3::from_shape_vec((nx, ny, nz), out_vec).unwrap()
}

/// 3D Binary Erosion
pub fn binary_erosion(input: &Array3<bool>, radius: usize) -> Array3<bool> {
    let shape = input.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    let offsets = get_ball_offsets(radius);

    let out_vec: Vec<bool> = (0..nx * ny * nz).into_par_iter().map(|idx| {
        let z = idx % nz;
        let y = (idx / nz) % ny;
        let x = idx / (ny * nz);
        
        // Erosion: true only if ALL neighbors in kernel are true.
        // So if ANY neighbor is false, result is false.
        // Wait, standard definition: 
        // A pixel is 1 if the structuring element centered at it is completely contained in the input set.
        // I.e., for all offsets in SE, input[coord + offset] must be 1.
        
        for &(dx, dy, dz) in &offsets {
            let ix = x as isize + dx;
            let iy = y as isize + dy;
            let iz = z as isize + dz;
            
            if ix >= 0 && ix < nx as isize &&
               iy >= 0 && iy < ny as isize &&
               iz >= 0 && iz < nz as isize {
                   if !input[[ix as usize, iy as usize, iz as usize]] {
                       return false;
                   }
            } else {
                // If the kernel goes outside, it's usually considered "background" (0).
                // So if we hit boundary, condition fails -> false.
                return false; 
            }
        }
        true
    }).collect();

    Array3::from_shape_vec((nx, ny, nz), out_vec).unwrap()
}
