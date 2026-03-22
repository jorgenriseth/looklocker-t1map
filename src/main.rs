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
mod morphology;
mod postprocess;
mod filters;

use model::LookLockerProblem;
use postprocess::PostProcessOptions;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    input: PathBuf,

    #[arg(short, long)]
    timestamps: PathBuf,

    #[arg(short, long)]
    output: PathBuf,
    
    /// Optional path to save the raw (unfiltered) T1 map.
    #[arg(long)]
    output_raw: Option<PathBuf>,

    /// Lower bound for valid T1 values (ms).
    #[arg(long, default_value_t = 0.0)]
    t1_low: f32,

    /// Upper bound for valid T1 values (ms).
    #[arg(long, default_value_t = 5500.0)]
    t1_high: f32,
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

    // 2. Thresholding pipeline (mri_facemask equivalent)
    let first_vol = mri_data.slice(s![.., .., .., 0]);
    
    // Step A: Triangle Threshold
    let triangle_thresh = masking::compute_triangle_threshold(&first_vol);
    println!("  Triangle threshold: {:.2}", triangle_thresh);
    let mut mask = first_vol.mapv(|v| v > triangle_thresh);
    
    // Step B: Fill holes
    mask = morphology::binary_fill_holes(&mask);
    
    // Step C: Gaussian Blur (sigma=5.0)
    let mask_float = mask.mapv(|b| if b { 1.0f32 } else { 0.0f32 });
    let blurred_mask = filters::gaussian_blur_3d(&mask_float, 5.0);
    
    // Step D: ISODATA Threshold
    let isodata_thresh = masking::compute_isodata_threshold(&blurred_mask.view());
    println!("  ISODATA threshold (on mask): {:.4}", isodata_thresh);
    
    mask = blurred_mask.mapv(|v| v > isodata_thresh);
    
    // 3. Valid voxels: (max_proj > 0) AND mask
    let mut t1_map = Array3::<f32>::from_elem((nx, ny, nz), f32::NAN);
    
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
        
        // If any time point is non-finite, skip fitting (matches Python behavior)
        if voxel_data.iter().any(|&v| !v.is_finite()) {
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
        
        // Use a high roof for raw data, but post-processing will clamp tighter.
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

    // Save raw output if requested
    if let Some(raw_path) = &args.output_raw {
        println!("Saving raw output...");
        let mut raw_data = t1_map.clone();
        raw_data.mapv_inplace(|v| if v.is_nan() { 0.0 } else { v });
        io::save_nifti(raw_path, raw_data.into_dyn(), &header)?;
    }

    // Post-processing
    println!("Starting post-processing...");
    let options = PostProcessOptions {
        t1_low: args.t1_low,
        t1_high: args.t1_high,
        ..Default::default()
    };

    postprocess::clean_t1_map(&mut t1_map, &mask, &options);

    println!("Saving post-processed output...");
    t1_map.mapv_inplace(|v| if v.is_nan() { 0.0 } else { v });
    let output_data = t1_map.into_dyn();
    io::save_nifti(&args.output, output_data, &header)?;

    println!("Complete.");
    Ok(())
}
