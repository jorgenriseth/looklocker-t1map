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
    let mut visited    = Array3::from_elem((nx, ny, nz), false);
    let mut queue      = VecDeque::new();

    let seed = |x: usize, y: usize, z: usize,
                    q:   &mut VecDeque<(usize, usize, usize)>,
                    vis: &mut Array3<bool>,
                    bg:  &mut Array3<bool>| {
        if !mask[[x, y, z]] && !vis[[x, y, z]] {
            vis[[x, y, z]] = true;
            bg[[x, y, z]]  = true;
            q.push_back((x, y, z));
        }
    };

    // Seed all six faces.
    for y in 0..ny { for z in 0..nz {
        seed(0,    y, z, &mut queue, &mut visited, &mut background);
        seed(nx-1, y, z, &mut queue, &mut visited, &mut background);
    }}
    for x in 0..nx { for z in 0..nz {
        seed(x, 0,    z, &mut queue, &mut visited, &mut background);
        seed(x, ny-1, z, &mut queue, &mut visited, &mut background);
    }}
    for x in 0..nx { for y in 0..ny {
        seed(x, y, 0,    &mut queue, &mut visited, &mut background);
        seed(x, y, nz-1, &mut queue, &mut visited, &mut background);
    }}

    // 26-connected flood fill.
    const NEIGHBORS: [(isize, isize, isize); 26] = [
        // 6 face
        (-1, 0, 0), (1, 0, 0), (0,-1, 0), (0, 1, 0), (0, 0,-1), (0, 0, 1),
        // 12 edge
        (-1,-1, 0), (-1, 1, 0), (1,-1, 0), (1, 1, 0),
        (-1, 0,-1), (-1, 0, 1), (1, 0,-1), (1, 0, 1),
        ( 0,-1,-1), ( 0,-1, 1), (0, 1,-1), (0, 1, 1),
        // 8 corner
        (-1,-1,-1), (-1,-1, 1), (-1, 1,-1), (-1, 1, 1),
        ( 1,-1,-1), ( 1,-1, 1), ( 1, 1,-1), ( 1, 1, 1),
    ];

    while let Some((cx, cy, cz)) = queue.pop_front() {
        for &(dx, dy, dz) in &NEIGHBORS {
            let ix = cx as isize + dx;
            let iy = cy as isize + dy;
            let iz = cz as isize + dz;
            if ix >= 0 && ix < nx as isize &&
               iy >= 0 && iy < ny as isize &&
               iz >= 0 && iz < nz as isize {
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
