# looklocker-t1map

A fast Rust tool for estimating T1 relaxation maps from Look-Locker MRI data. It takes a 4D NIfTI file and trigger timestamps as input and produces a cleaned T1 map (in ms) as output.

## Download

Pre-built binaries for Linux, macOS, and Windows are available on the [Releases](../../releases) page. Download the archive for your platform, extract it, and run the binary directly — no installation required.

## Container (Docker / Singularity)

A minimal container image is published to the GitHub Container Registry on every release.

### Docker

Pull and run the pre-built image:

```bash
docker pull ghcr.io/jorgenriseth/looklocker-t1map:latest

docker run --rm \
  -v "$(pwd)":/data \
  ghcr.io/jorgenriseth/looklocker-t1map:latest \
  --input  /data/sub-01_acq-looklocker_IRT1.nii.gz \
  --timestamps /data/sub-01_acq-looklocker_IRT1_trigger_times.txt \
  --output /data/t1_map.nii.gz
```

The `-v "$(pwd)":/data` flag mounts your current directory into the container at `/data`. Adjust the mount as needed.

To build the image locally:

```bash
docker build -t looklocker-t1map .
```

### Singularity / Apptainer

Pull the Docker image as a Singularity image file (`.sif`):

```bash
singularity pull looklocker-t1map.sif docker://ghcr.io/jorgenriseth/looklocker-t1map:latest
```

Run it:

```bash
singularity run looklocker-t1map.sif \
  --input  sub-01_acq-looklocker_IRT1.nii.gz \
  --timestamps sub-01_acq-looklocker_IRT1_trigger_times.txt \
  --output t1_map.nii.gz
```

Singularity bind-mounts your current directory automatically, so paths relative to `$PWD` work without extra flags.

## Build from source

Requires [Rust](https://rustup.rs/).

```bash
cargo build --release
# binary at: target/release/looklocker-t1map
```

## Usage

```bash
looklocker-t1map \
  --input  data/sub-01_acq-looklocker_IRT1.nii.gz \
  --timestamps data/sub-01_acq-looklocker_IRT1_trigger_times.txt \
  --output output/t1_map.nii.gz
```

### All options

| Argument | Description | Default |
|:---|:---|:---|
| `--input`, `-i` | 4D NIfTI file (x, y, z, t) | required |
| `--timestamps`, `-t` | Text file of trigger times in milliseconds | required |
| `--output`, `-o` | Path for the post-processed T1 map | required |
| `--output-raw` | Path for the raw (unfiltered) T1 map | — |
| `--t1-low` | Lower bound for valid T1 values (ms) | `0.0` |
| `--t1-high` | Upper bound for valid T1 values (ms) | `5500.0` |

The timestamps file should contain whitespace-separated values, one per volume, in milliseconds.

---

## Pipeline

### 1. Mask generation

A brain mask is computed from the data to restrict fitting to relevant voxels:

1. **Triangle threshold** on the first volume to produce an initial binary mask.
2. **Hole filling** (26-connectivity flood fill) to capture internal structures.
3. **Gaussian blur** (σ = 5.0) of the binary mask.
4. **ISODATA threshold** on the blurred mask to define the final tight brain boundary.

Voxels must also have a positive max signal across time to be included.

### 2. Voxel-wise T1 fitting

For each masked voxel, the Look-Locker signal model is fitted using the Levenberg-Marquardt algorithm:

$$M(t) = \left| M_0 \cdot \left(1 - (1 + \alpha^2)\, e^{-R^2 t}\right) \right|$$

The time series is normalised to its maximum before fitting. The apparent relaxation time from the Look-Locker sequence is $T_1^* = 1/R^2$, which is shorter than the true $T_1$ due to the repeated RF pulses. The Look-Locker correction is applied analytically: the steady-state factor $(1 + \alpha^2)$ in the model means $T_1 = (\alpha^2) \cdot T_1^* = (\alpha / R)^2$. Fitting runs in parallel across all CPU cores.

### 3. Post-processing

The raw T1 map is cleaned in three steps:

1. **Masking** — voxels outside the brain mask are set to zero.
2. **Outlier removal** — voxels outside `[t1_low, t1_high]` are removed.
3. **Hole filling** — remaining gaps inside the mask are filled iteratively using a Gaussian-weighted average of neighbouring finite voxels (σ = 1.0).
