# Look-Locker T1-Map Estimation (Rust)

A high-performance Rust implementation of the Look-Locker T1-map estimation
pipeline for NIfTI MRI data. This tool processes 4D MRI data to estimate T1
relaxation times using a non-linear least squares fit, complete with robust
masking and post-processing utilities.

## Features

- **Fast & Parallel**: Utilizes all available CPU cores for voxel-wise curve
  fitting (processing millions of voxels in seconds).
- **Robust Fitting**: Implements the Look-Locker model using the
  Levenberg-Marquardt algorithm.
- **Advanced Post-Processing**: Includes morphological cleaning
  (erosion/dilation), outlier removal, and iterative hole filling to produce
  high-quality T1 maps.
- **Native NIfTI Support**: Reads and writes NIfTI files directly, preserving
  affine transformations.
- **Flexible CLI**: Customizable thresholds and output options.

## Installation & Building

Ensure you have [Rust and Cargo installed](https://rustup.rs/).

1. **Clone the repository** (if applicable) or navigate to the project
    directory:

    ```bash
    cd looklocker-t1map-rs
    ```

2. **Build the project in release mode**:

    ```bash
    cargo build --release
    ```

    The compiled binary will be located at `target/release/looklocker-t1map-rs`.

## Usage

Run the tool using the compiled binary.

### Basic Command

```bash
./target/release/looklocker-t1map-rs \
  --input <path/to/mri_data.nii.gz> \
  --timestamps <path/to/timestamps.txt> \
  --output <path/to/output_t1map.nii.gz>
```

### Full Options

```bash
./target/release/looklocker-t1map-rs \
  --input data/sub-01_ses-01_acq-looklocker_IRT1.nii.gz \
  --timestamps data/sub-01_ses-01_acq-looklocker_IRT1_trigger_times.txt \
  --output output/t1_map_clean.nii.gz \
  --output-raw output/t1_map_raw.nii.gz \
  --t1-low 0 \
  --t1-high 5500
```

### Arguments

| Argument             | Description                                                                      | Default  |
| :------------------- | :------------------------------------------------------------------------------- | :------- |
| `--input`, `-i`      | Path to the input 4D NIfTI MRI file (x, y, z, t).                                | Required |
| `--timestamps`, `-t` | Path to the text file containing trigger times (in ms).                          | Required |
| `--output`, `-o`     | Path to save the final **post-processed** T1 map.                                | Required |
| `--output-raw`       | Optional path to save the raw (unfiltered) T1 map.                               | None     |
| `--t1-low`           | Lower bound for valid T1 values (ms). Values below this are treated as outliers. | `0.0`    |
| `--t1-high`          | Upper bound for valid T1 values (ms). Values above this are treated as outliers. | `5500.0` |

## Input Formats

- **MRI Data**: 4D NIfTI file (`.nii` or `.nii.gz`) where the 4th dimension
  represents time.
- **Timestamps**: A plain text file with whitespace-separated values
  representing the trigger times in **milliseconds**. The number of timestamps
  must match the number of time points in the MRI data.

## Post-Processing Logic

The tool automatically applies the following steps to clean the T1 map:

1. **Mask Generation**: Identifies the largest connected component of the
    signal.
2. **Morphological Cleaning**: Removes small holes, dilates the mask
    (radius 10) to include boundaries, and then erodes (radius 13) to remove
    edge artifacts.
3. **Outlier Removal**: Sets voxels outside the mask or outside the valid T1
    range (`[t1_low, t1_high]`) to NaN.
4. **Hole Filling**: Iteratively fills internal gaps (NaNs) using a
    Gaussian-weighted local average.

## License

[MIT](LICENSE) (or applicable license)
