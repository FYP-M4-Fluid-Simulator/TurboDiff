import jax.numpy as jnp
from turbodiff.core.cst import create_cst_sdf
from turbodiff.core.utils import create_solid_mask


def draw_mask(mask):
    import numpy as np

    m = np.asarray(mask)
    for i in range(m.shape[0]):
        row = ""
        for j in range(m.shape[1]):
            if m[i, j] > 0.5:
                row += "#"
            else:
                row += "."
        print(row)


def test_cst_airfoil():
    print("Testing CST Airfoil generation...")
    # Symmetrical section resembling NACA 0012
    # For a symmetrical airfoil order 4 (5 weights)
    w_u = [0.15, 0.15, 0.15, 0.15, 0.15]
    w_l = [-0.15, -0.15, -0.15, -0.15, -0.15]

    sdf_fn = create_cst_sdf(
        weights_upper=w_u,
        weights_lower=w_l,
        chord=0.6,
        cell_size=0.01,
        center_x=0.2,
        center_y=0.2,
    )

    mask = create_solid_mask((40, 100), boundary=0, sdf_fn=sdf_fn)

    print("Mask max:", jnp.max(mask))
    print("Mask sum:", jnp.sum(mask))

    draw_mask(mask)

    print("Success!")


if __name__ == "__main__":
    test_cst_airfoil()
