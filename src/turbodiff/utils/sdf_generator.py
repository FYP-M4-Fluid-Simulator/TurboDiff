import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from matplotlib.path import Path





# ---------------- SDF QUERY ----------------

def compute_sdf_at_point(x, y, path, kdtree):
    # Nearest surface distance
    dist, _ = kdtree.query([x, y], k=1)

    # Inside / outside test
    is_inside = path.contains_point((x, y))

    # return negative distance if inside
    return -dist if is_inside else dist


# ---------------- AIRFOIL HELPERS ----------------

def load_dat_file(filename):
    coords = []
    with open(filename, "r") as f:
        for line in f:
            try:
                parts = line.strip().split()
                if len(parts) >= 2:
                    coords.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    return coords


# ---------------- MAIN ----------------


def prepare_geometry(grid_height, grid_width, dat_filepath, chord_length, offset_x, offset_y):
    # Grid
    H, W = grid_height, grid_width
    grid_sdf = np.zeros((H, W))

    # Load airfoil
    print("Generating Airfoil Geometry...")
    coords = load_dat_file(dat_filepath)

        # 1. Load points
    points = np.array(coords)

    # 2. Scale
    points *= chord_length

    # 3. Translate
    offset = (offset_x, offset_y)
    points[:, 0] += offset[0]
    points[:, 1] += offset[1]

    # 4. Acceleration structures
    path = Path(points)
    kdtree = cKDTree(points)


    return grid_sdf, path, kdtree

if __name__ == "__main__":
    # Grid dimensions
    H, W = 200, 400
    grid_sdf, path, kdtree = prepare_geometry(
        grid_height=H,
        grid_width=W,
        dat_filepath="Airfoils\\RAE2822.dat",
        chord_length=120,
        offset_x=50,
        offset_y=40
    )
    

        # Compute SDF
    print("Computing SDF for entire grid...")
    for row in range(H):
        for col in range(W):
            grid_sdf[row, col] = compute_sdf_at_point(col, row, path, kdtree)


    print(grid_sdf[:10, :10])
   

    # Visualize
    plt.figure(figsize=(10, 5))
    limit = np.max(np.abs(grid_sdf))

    plt.imshow(
        grid_sdf,
        origin="lower",
        cmap="RdBu",
        vmin=-limit,
        vmax=limit
    )
    plt.colorbar(label="Euclidean Distance")
    plt.contour(grid_sdf, levels=[0], colors="black", linewidths=2)

    plt.title("Complete Airfoil SDF\n, Chord 120px")
    plt.xlabel("X")
    plt.ylabel("Y")
    # plt.axis("equal")
    plt.show()

    print("Done.")
