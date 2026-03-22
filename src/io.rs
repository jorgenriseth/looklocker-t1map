use anyhow::{Context, Result};
use ndarray::ArrayD;
use nifti::{IntoNdArray, NiftiHeader, NiftiObject, ReaderOptions};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub fn load_timestamps(path: &Path) -> Result<Vec<f64>> {
    let file =
        File::open(path).with_context(|| format!("Failed to open timestamps file: {:?}", path))?;
    let reader = BufReader::new(file);
    let mut timestamps = Vec::new();

    for line in reader.lines() {
        let line = line?;
        // Assuming one timestamp per line or space separated
        for part in line.split_whitespace() {
            let t: f64 = part
                .parse()
                .with_context(|| format!("Failed to parse timestamp: {}", part))?;
            timestamps.push(t);
        }
    }

    // Convert ms to s as in Python script (divide by 1000 if values are large, but python script does explicit division)
    // The python script says: `time = np.loadtxt(timestamps) / 1000`
    // So we should divide by 1000.
    let timestamps_s: Vec<f64> = timestamps.iter().map(|&t| t / 1000.0).collect();

    Ok(timestamps_s)
}

pub type NiftiData = (ArrayD<f32>, nifti::NiftiHeader);

pub fn load_nifti(path: &Path) -> Result<NiftiData> {
    let obj = ReaderOptions::new()
        .read_file(path)
        .with_context(|| format!("Failed to read NIfTI file: {:?}", path))?;
    let header = obj.header().clone();
    let volume = obj
        .into_volume()
        .into_ndarray::<f32>()
        .with_context(|| "Failed to convert volume to ndarray")?;

    Ok((volume, header))
}

pub fn save_nifti(path: &Path, data: ArrayD<f32>, reference_header: &NiftiHeader) -> Result<()> {
    // We use the reference header to preserve affine and other metadata.
    // However, nifti-rs writing support is sometimes limited or requires specific handling.
    // For now, we will try to write using the standard writer if available or reconstruct.
    // The nifti crate allows writing, but we might need to be careful with extensions.

    // Simplest way with nifti crate 0.17:
    // nifti::WriterOptions::new(path).write_nifti(&data) ... but we need to set the affine.

    // Actually, nifti-rs writing usually takes a header.

    use nifti::writer::WriterOptions;

    WriterOptions::new(path)
        .reference_header(reference_header)
        .write_nifti(&data)
        .with_context(|| format!("Failed to write NIfTI file: {:?}", path))?;

    Ok(())
}
