use clap::Parser;
use std::path::PathBuf;
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array3, s};
use rayon::prelude::*;
use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::{DVector, Vector3};

mod io;
mod model;
mod masking;

use model::LookLockerProblem;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    input: PathBuf,

    #[arg(short, long)]
    timestamps: PathBuf,

    #[arg(short, long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Loading data...");
    let timestamps = io::load_timestamps(&args.timestamps)?;
    let (mri_data, header) = io::load_nifti(&args.input)?;
    
    let shape = mri_data.shape();
    if shape.len() != 4 {
        anyhow::bail!("Input MRI data must be 4D (x, y, z, t). Got shape: {:?}", shape);
    }
    let (nx, ny, nz, nt) = (shape[0], shape[1], shape[2], shape[3]);

    if nt != timestamps.len() {
        anyhow::bail!("Number of time points in MRI ({}) does not match timestamps ({})", nt, timestamps.len());
    }

    println!("Computing mask...");
    // 1. Max projection along time axis
    let mut max_proj = Array3::<f32>::zeros((nx, ny, nz));
    
    // Iterate over spatial dimensions and find max time value
    // We can use Zip or just nested loops. Zip is faster/cleaner with ndarray.
    // Or we can slice.
    // Let's use simple nested loops for clarity or Zip if possible.
    // ndarray doesn't have a direct max_axis for float.
    
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let mut max_val = f32::NEG_INFINITY;
                for t in 0..nt {
                    let val = mri_data[[x, y, z, t]];
                    if val > max_val {
                        max_val = val;
                    }
                }
                max_proj[[x, y, z]] = max_val;
            }
        }
    }

    // 2. Threshold > 0 and masking on first volume (Python logic: mri_facemask(D[..., 0]))
    // Python's mri_facemask usually operates on structural info.
    // The python script uses `mri_facemask(D[..., 0])`.
    // Let's compute mask on first volume > 0.
    
    let first_vol = mri_data.slice(s![.., .., .., 0]);
    let mut mask_input = Array3::<bool>::from_elem((nx, ny, nz), false);
    for ((x, y, z), &val) in first_vol.indexed_iter() {
        if val > 0.0 {
            mask_input[[x, y, z]] = true;
        }
    }
    
    let mask = masking::get_largest_component_mask(&mask_input);
    
    // 3. Valid voxels: (max_proj > 0) AND mask
    // We'll just iterate over the mask.
    
    let mut t1_map = Array3::<f32>::from_elem((nx, ny, nz), f32::NAN);
    
    // Flatten indices for parallel iteration
    let mut indices = Vec::new();
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                if mask[[x, y, z]] && max_proj[[x, y, z]] > 0.0 {
                    indices.push((x, y, z));
                }
            }
        }
    }

    println!("Fitting {} voxels...", indices.len());
    let pb = ProgressBar::new(indices.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("#>-"));

    let timestamps_vec = DVector::from_vec(timestamps.clone());
    
    // We can't share ProgressBar across threads easily without overhead or specific crates.
    // `indicatif` works with `rayon` via `par_iter` if we wrap it, but standard approach is periodic update or just simple finish.
    // For simplicity, we can use `pb.inc` in a rough way or just print status.
    // `indicatif` has `ParallelProgressIterator` trait if enabled? No, usually requires `indicatif` features.
    // We'll skip granular progress bar update inside rayon for now to avoid complexity or use a scoped thread if needed.
    // Or we can just use `pb.inc(1)` if we use `par_iter().for_each`. Rayon `for_each` takes a closure. `pb` needs to be thread-safe. `ProgressBar` is thread-safe.
    
    // We use par_iter to map indices to results.
    let results: Vec<((usize, usize, usize), f32)> = indices.par_iter().map(|&(x, y, z)| {
        // Extract time series
        let mut voxel_data = Vec::with_capacity(nt);
        let mut max_v = f32::NEG_INFINITY;
        let mut min_v = f32::INFINITY;
        let mut min_idx = 0;
        
        for t in 0..nt {
            let v = mri_data[[x, y, z, t]];
            voxel_data.push(v as f64);
            if v > max_v { max_v = v; }
            if v < min_v { 
                min_v = v; 
                min_idx = t;
            }
        }
        
        let max_val = max_v as f64;
        if max_val <= 0.0 {
            pb.inc(1);
            return ((x, y, z), f32::NAN);
        }
        
        let y_data: Vec<f64> = voxel_data.iter().map(|&v| v / max_val).collect();
        let y_vec = DVector::from_vec(y_data);

        // Initial guess
        let t_min = timestamps[min_idx];
        let t1_est = t_min / (1.0 + 1.25f64).ln();
        
        let x1 = 1.0;
        let x2 = 1.25f64.sqrt();
        let x3 = if t1_est > 0.0 { (1.0 / t1_est).sqrt() } else { 1.0 }; 
        
        let p0 = Vector3::new(x1, x2, x3);
        
        let problem = LookLockerProblem {
            t: timestamps_vec.clone(),
            y: y_vec,
            p: p0,
        };
        
        let (result, _report) = LevenbergMarquardt::new().minimize(problem);
        
        let p_opt = result.p;
        let x2_opt = p_opt[1];
        let x3_opt = p_opt[2];
        
        let t1 = if x3_opt.abs() > 1e-9 {
            (x2_opt / x3_opt).powi(2) * 1000.0
        } else {
            f32::NAN as f64
        };
        
        let t1_roof = 10000.0;
        let t1_final = if t1.is_nan() { f32::NAN } else { t1.min(t1_roof) as f32 };
        
        pb.inc(1);
        ((x, y, z), t1_final)
    }).collect();

    pb.finish_with_message("Done fitting");

    // Fill map
    for ((x, y, z), val) in results {
        t1_map[[x, y, z]] = val;
    }

    println!("Saving output...");
    // Convert to ArrayD
    let output_data = t1_map.into_dyn();
    io::save_nifti(&args.output, output_data, &header)?;

    println!("Complete.");
    Ok(())
}
