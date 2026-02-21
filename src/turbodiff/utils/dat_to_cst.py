from lib.airfoil_generation import class_shape_transformation
import numpy as np

from turbodiff.utils._evaluate_error import measure_closeness


def CST_fitting(input_path, output_path, Bernstein_order, _R_le):
    """
    args:
        - input_path: the path of the input .dat file (pointwise style)
        - output_path: the path to save the output .dat file (pointwise style)
        - Bernstein_order: the order of Bernstein polynomials used in CST fitting (e.g., 4, 6, 8)

    returns:
        - upperCoefficients: the estimated CST coefficients for the upper surface
        - lowerCoefficients: the estimated CST coefficients for the lower surface
        - accuracy: the fitting accuracy (e.g., relative Euclidean distance between original and fitted coordinates)
        - message: a success message indicating the file was converted successfully


    Initial settings according to different type of airfoils
    There are only two sets needed to be modified
    (filename   and   leading edge radius)
    """

    fitting_airfoil = class_shape_transformation.CST(input_path, Bernstein_order)

    # _x, _y are the x , y values of the input airfoil
    # _x_id is the index of point to split the upper and lower surface (the starting point)
    # _size is the no. of points of the input airfoil (number of x points)

    (_x, _y, _x_id, _size) = fitting_airfoil.loaddata()

    (x_up, y_up, x_low, y_low) = fitting_airfoil.datasplit(_x, _y, _x_id, _size)

    # R_le = 0.008496  # RAE2822 leading edge radius
    # R_le = 0.0125  # NACA0012 leading edge radius
    R_le = _R_le  # NACA0012 leading edge radius

    """
    up_surface fitting
    """

    # alpha_te: how much the upper surface of trailing edge is bent
    # Y_te is the last point of Y in the upper surface
    (alpha_te, Y_te) = fitting_airfoil.compute_half_alpha_thickness_te(x_up, y_up)

    C_up = fitting_airfoil.classfunction(x_up)
    B_up = fitting_airfoil.bernstein(x_up)

    a_up = fitting_airfoil.comp_initial_control_points(R_le, Y_te, alpha_te, x_up, y_up)

    # --- ADD THIS LINE ---
    a_up = np.squeeze(a_up)
    # ---------------------

    S_up = fitting_airfoil.shapefunction(a_up, B_up)

    y_CST_up = fitting_airfoil.CST_fitting(C_up, S_up, x_up, Y_te)

    """
    low_surface fitting 
    First we tranform the y value in y-positive direction
    Finally we transform the y_CST_low value in the negative direction
    """
    y_low = -y_low[:]

    alpha_te, Y_te = fitting_airfoil.compute_half_alpha_thickness_te(x_low, y_low)
    C_low = fitting_airfoil.classfunction(x_low)  # function calls
    B_low = fitting_airfoil.bernstein(x_low)  # function calls
    a_low = fitting_airfoil.comp_initial_control_points(
        R_le, Y_te, alpha_te, x_low, y_low
    )  # function calls

    # --- ADD THIS LINE ---
    a_low = np.squeeze(a_low)
    # ---------------------

    S_low = fitting_airfoil.shapefunction(a_low, B_low)
    y_CST_low = fitting_airfoil.CST_fitting(C_low, S_low, x_low, Y_te)
    y_CST_low = -y_CST_low[:, :]
    y_low = -y_low[:]

    # """
    # save the fitting result to a pointwise style .dat file to do mesh generation
    # """
    # y_fit = np.empty((_size, 1))

    # y_fit[0:len(x_up), 0] = y_CST_up[::-1, 0]
    # for i in range(len(x_up), _size):
    #     y_fit[i, 0] = y_CST_low[ -len(x_low)+1, 0]

    # fitting_airfoil.datsave(output_path, _x, y_fit[:, 0])

    """
    save the fitting result to a pointwise style .dat file
    """
    y_fit = np.empty((_size, 1))

    # 1. Fill Upper Surface (Reverse order as before)
    y_fit[0 : len(x_up), 0] = y_CST_up[::-1, 0]

    # 2. Fill Lower Surface (SAFE METHOD)
    # We use a separate counter 'k' for the lower surface array.
    # We start k=1 because the Leading Edge (index 0) is already covered by the Upper Surface.
    k = 1

    for i in range(len(x_up), _size):
        if k < len(y_CST_low):
            y_fit[i, 0] = y_CST_low[k, 0]
            k += 1
        else:
            # OPTIONAL: If we run out of points (rare rounding error case),
            # repeat the last known point to prevent crashing.
            y_fit[i, 0] = y_CST_low[-1, 0]

            # save it later in the cloud / db
    # fitting_airfoil.datsave(output_path, _x, y_fit[:, 0])

    # --- CALL THE FUNCTION HERE ---
    # _x: original x, _y: original y
    # _x: predicted x (CST uses same x), y_fit: predicted y
    accuracy = measure_closeness(_x, _y, _x, y_fit)
    print("------------------------------------------------")
    print(f"CST Fitting Accuracy (Relative Euclidean): {accuracy:.4f}%")
    print("------------------------------------------------")

    # Extract just the filename (e.g., "naca0012.dat")
    # name_only = os.path.basename(input_path)

    # # Open file in Append mode ('a') and write the result
    # with open(REPORT_FILE, "a") as f:
    #     f.write(f"{name_only:<25} \t {accuracy:.6f}%\n")
    # # ------------------------------

    """
    error evaluating
    # """
    # error_up = fitting_airfoil.error_eval(y_up, y_CST_up[:, 0])
    # error_low = fitting_airfoil.error_eval(y_low, y_CST_low[:, 0])

    # lower points are opposite
    a_low = -a_low
    return {
        "upperCoefficients": a_up.tolist(),
        "lowerCoefficients": a_low.tolist(),
        "accuracy": accuracy,
        "message": "File converted successfully",
    }
