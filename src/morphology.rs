use ndarray::Array3;
use std::collections::VecDeque;

/// Fills holes in a binary mask by flood-filling the background from all image borders.
///
/// Uses 26-connectivity, matching scipy.ndimage.binary_fill_holes default behaviour
/// (generate_binary_structure(3, 3)).  6-connectivity would block diagonal passages and
/// incorrectly fill more regions, producing a larger mask.
pub fn binary_fill_holes(mask: &Array3<bool>) -> Array3<bool> {
    let shape = mask.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    let mut background = Array3::from_elem((nx, ny, nz), false);
    let mut visited = Array3::from_elem((nx, ny, nz), false);
    let mut queue = VecDeque::new();

    let seed = |x: usize,
                y: usize,
                z: usize,
                q: &mut VecDeque<(usize, usize, usize)>,
                vis: &mut Array3<bool>,
                bg: &mut Array3<bool>| {
        if !mask[[x, y, z]] && !vis[[x, y, z]] {
            vis[[x, y, z]] = true;
            bg[[x, y, z]] = true;
            q.push_back((x, y, z));
        }
    };

    // Seed all six faces.
    for y in 0..ny {
        for z in 0..nz {
            seed(0, y, z, &mut queue, &mut visited, &mut background);
            seed(nx - 1, y, z, &mut queue, &mut visited, &mut background);
        }
    }
    for x in 0..nx {
        for z in 0..nz {
            seed(x, 0, z, &mut queue, &mut visited, &mut background);
            seed(x, ny - 1, z, &mut queue, &mut visited, &mut background);
        }
    }
    for x in 0..nx {
        for y in 0..ny {
            seed(x, y, 0, &mut queue, &mut visited, &mut background);
            seed(x, y, nz - 1, &mut queue, &mut visited, &mut background);
        }
    }

    // 26-connected flood fill.
    const NEIGHBORS: [(isize, isize, isize); 26] = [
        // 6 face
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
        // 12 edge
        (-1, -1, 0),
        (-1, 1, 0),
        (1, -1, 0),
        (1, 1, 0),
        (-1, 0, -1),
        (-1, 0, 1),
        (1, 0, -1),
        (1, 0, 1),
        (0, -1, -1),
        (0, -1, 1),
        (0, 1, -1),
        (0, 1, 1),
        // 8 corner
        (-1, -1, -1),
        (-1, -1, 1),
        (-1, 1, -1),
        (-1, 1, 1),
        (1, -1, -1),
        (1, -1, 1),
        (1, 1, -1),
        (1, 1, 1),
    ];

    while let Some((cx, cy, cz)) = queue.pop_front() {
        for &(dx, dy, dz) in &NEIGHBORS {
            let ix = cx as isize + dx;
            let iy = cy as isize + dy;
            let iz = cz as isize + dz;
            if ix >= 0
                && ix < nx as isize
                && iy >= 0
                && iy < ny as isize
                && iz >= 0
                && iz < nz as isize
            {
                let (ux, uy, uz) = (ix as usize, iy as usize, iz as usize);
                if !mask[[ux, uy, uz]] && !visited[[ux, uy, uz]] {
                    visited[[ux, uy, uz]] = true;
                    background[[ux, uy, uz]] = true;
                    queue.push_back((ux, uy, uz));
                }
            }
        }
    }

    // Every voxel that is NOT background is either inside the original mask or a filled hole.
    background.mapv(|b| !b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn fill_holes_no_holes_unchanged() {
        let mask = Array3::from_elem((3, 3, 3), true);
        let filled = binary_fill_holes(&mask);
        assert!(filled.iter().all(|&v| v));
    }

    #[test]
    fn fill_holes_all_false_unchanged() {
        // All-false: all border voxels are seeded as background, flood fills everything.
        // Result: all background → mapv(!b) → all false.
        let mask = Array3::from_elem((3, 3, 3), false);
        let filled = binary_fill_holes(&mask);
        assert!(filled.iter().all(|&v| !v));
    }

    #[test]
    fn fill_holes_interior_hole_is_filled() {
        // 5×5×5 solid mask with one interior voxel set false.
        // All border voxels are true → not seeded. Interior hole not reachable → filled.
        let mut mask = Array3::from_elem((5, 5, 5), true);
        mask[[2, 2, 2]] = false;
        let filled = binary_fill_holes(&mask);
        assert!(filled[[2, 2, 2]], "interior hole should be filled");
        assert!(filled[[0, 0, 0]]);
    }

    #[test]
    fn fill_holes_border_background_not_filled() {
        // Background voxel on the border must NOT become true (it's reachable from seed).
        let mut mask = Array3::from_elem((5, 5, 5), true);
        mask[[0, 2, 2]] = false;
        let filled = binary_fill_holes(&mask);
        assert!(!filled[[0, 2, 2]], "border background should stay false");
    }

    #[test]
    fn fill_holes_closed_3d_shell() {
        // 7×7×7: only the 6 outer faces are true; interior 5×5×5 is false.
        // All array-border voxels are true → not seeded → no flood fill runs.
        // Interior 125 voxels are unreachable → filled (result = true).
        let n = 7usize;
        let mut mask = Array3::from_elem((n, n, n), false);
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    if x == 0 || x == n - 1 || y == 0 || y == n - 1 || z == 0 || z == n - 1 {
                        mask[[x, y, z]] = true;
                    }
                }
            }
        }
        let filled = binary_fill_holes(&mask);
        for x in 1..n - 1 {
            for y in 1..n - 1 {
                for z in 1..n - 1 {
                    assert!(filled[[x, y, z]], "interior [{x},{y},{z}] not filled");
                }
            }
        }
        assert!(filled[[0, 0, 0]]);
    }
}
