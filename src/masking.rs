use ndarray::Array3;
use std::collections::VecDeque;

pub fn get_largest_component_mask(input: &Array3<bool>) -> Array3<bool> {
    let shape = input.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    let mut visited = Array3::from_elem((nx, ny, nz), false);
    let mut labels = Array3::from_elem((nx, ny, nz), 0usize);
    let mut current_label = 0;
    let mut label_counts = Vec::new();

    // 6-connectivity neighbors
    let neighbors = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1),
    ];

    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                if input[[x, y, z]] && !visited[[x, y, z]] {
                    current_label += 1;
                    let mut count = 0;
                    let mut queue = VecDeque::new();
                    queue.push_back((x, y, z));
                    visited[[x, y, z]] = true;
                    labels[[x, y, z]] = current_label;
                    count += 1;

                    while let Some((cx, cy, cz)) = queue.pop_front() {
                        for &(dx, dy, dz) in &neighbors {
                            let nx_idx = cx as isize + dx;
                            let ny_idx = cy as isize + dy;
                            let nz_idx = cz as isize + dz;

                            if nx_idx >= 0 && nx_idx < nx as isize &&
                               ny_idx >= 0 && ny_idx < ny as isize &&
                               nz_idx >= 0 && nz_idx < nz as isize {
                                let (uix, uiy, uiz) = (nx_idx as usize, ny_idx as usize, nz_idx as usize);
                                if input[[uix, uiy, uiz]] && !visited[[uix, uiy, uiz]] {
                                    visited[[uix, uiy, uiz]] = true;
                                    labels[[uix, uiy, uiz]] = current_label;
                                    queue.push_back((uix, uiy, uiz));
                                    count += 1;
                                }
                            }
                        }
                    }
                    label_counts.push((current_label, count));
                }
            }
        }
    }

    if label_counts.is_empty() {
        return Array3::from_elem((nx, ny, nz), false);
    }

    // Find label with max count
    label_counts.sort_by_key(|k| k.1);
    let (largest_label, _) = label_counts.last().unwrap();
    
    // Create mask for largest label
    let mut mask = Array3::from_elem((nx, ny, nz), false);
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                if labels[[x, y, z]] == *largest_label {
                    mask[[x, y, z]] = true;
                }
            }
        }
    }
    
    mask
}
