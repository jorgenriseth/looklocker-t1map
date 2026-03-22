use ndarray::ArrayView3;

/// Computes the triangle threshold, matching scikit-image's threshold_triangle exactly:
/// 256 bins, flip logic, normalised perpendicular-distance formula, bin-centre return value.
pub fn compute_triangle_threshold(data: &ArrayView3<f32>) -> f32 {
    let num_bins = 256usize;

    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;

    for &v in data {
        if v.is_finite() {
            if v < min_val { min_val = v; }
            if v > max_val { max_val = v; }
        }
    }

    if min_val >= max_val {
        return min_val;
    }

    // Build histogram over [min_val, max_val] with num_bins bins.
    let bin_width = (max_val - min_val) / num_bins as f32;
    let mut hist = vec![0usize; num_bins];
    for &v in data {
        if v.is_finite() {
            let bin = ((v - min_val) / bin_width).floor() as usize;
            hist[bin.min(num_bins - 1)] += 1;
        }
    }

    // Peak, first and last non-zero bins.
    let arg_peak = hist.iter().enumerate().max_by_key(|&(_, &c)| c).unwrap().0;
    let peak_height = hist[arg_peak] as f64;
    let arg_low  = hist.iter().position(|&c| c > 0).unwrap_or(0);
    let arg_high = hist.iter().rposition(|&c| c > 0).unwrap_or(num_bins - 1);

    if arg_low == arg_high {
        return min_val + (arg_low as f32 + 0.5) * bin_width;
    }

    // Flip so the peak is always on the right (left tail becomes the search side).
    // Matches scikit-image: flip = (arg_peak - arg_low) < (arg_high - arg_peak)
    let flip = (arg_peak - arg_low) < (arg_high - arg_peak);
    let (hist_w, arg_low_w, arg_peak_w) = if flip {
        let h: Vec<usize> = hist.iter().rev().copied().collect();
        (h, num_bins - arg_high - 1, num_bins - arg_peak - 1)
    } else {
        (hist.clone(), arg_low, arg_peak)
    };

    // Triangle line from (0, 0) to (width, peak_height) in the working histogram.
    // Maximise length = peak_height_n * x - width_n * y  (normalised signed distance).
    let width = (arg_peak_w - arg_low_w) as f64;
    let norm = (peak_height * peak_height + width * width).sqrt();
    let peak_height_n = peak_height / norm;
    let width_n = width / norm;

    let mut max_length = f64::NEG_INFINITY;
    let mut arg_level_w = arg_low_w;

    for i in 0..(arg_peak_w - arg_low_w) {
        let y = hist_w[i + arg_low_w] as f64;
        let length = peak_height_n * i as f64 - width_n * y;
        if length > max_length {
            max_length = length;
            arg_level_w = i + arg_low_w;
        }
    }

    // Unflip and return bin centre — matches scikit-image's bin_centers[arg_level].
    let arg_level = if flip { num_bins - arg_level_w - 1 } else { arg_level_w };
    min_val + (arg_level as f32 + 0.5) * bin_width
}

/// Computes the ISODATA threshold (iterative intermeans), matching scikit-image's
/// threshold_isodata: initial threshold at (min + max) / 2, iterate until convergence.
pub fn compute_isodata_threshold(data: &ArrayView3<f32>) -> f32 {
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;

    for &v in data {
        if v.is_finite() {
            if v < min_val { min_val = v; }
            if v > max_val { max_val = v; }
        }
    }

    if min_val >= max_val { return min_val; }

    // scikit-image initialises at (min + max) / 2, not the mean.
    let mut threshold = (min_val + max_val) / 2.0;
    let tolerance = 1e-4 * (max_val - min_val);

    for _ in 0..100 {
        let mut sum_lower = 0.0f64;
        let mut count_lower = 0usize;
        let mut sum_upper = 0.0f64;
        let mut count_upper = 0usize;

        for &v in data {
            if v.is_finite() {
                if v <= threshold {
                    sum_lower += v as f64;
                    count_lower += 1;
                } else {
                    sum_upper += v as f64;
                    count_upper += 1;
                }
            }
        }

        let mean_lower = if count_lower > 0 { sum_lower / count_lower as f64 } else { min_val as f64 };
        let mean_upper = if count_upper > 0 { sum_upper / count_upper as f64 } else { max_val as f64 };
        let new_threshold = ((mean_lower + mean_upper) / 2.0) as f32;

        if (new_threshold - threshold).abs() < tolerance {
            return new_threshold;
        }
        threshold = new_threshold;
    }

    threshold
}
