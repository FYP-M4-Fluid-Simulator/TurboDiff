import numpy as np
import math as mh


class CST:
    def __init__(self, filename, Bernstein_order):
        self.filename = filename
        self.Bernstein_order = Bernstein_order

    def loaddata(self):
        """
            Load airfoil data from .dat file.
        Returns:
            X array

            Y array

            index of leading edge

            total number of points.
        """

        filename = self.filename

        with open(filename, "r") as f:
            next(f)  # skip the first line
            lines = f.readlines()
            X, Y, Z = [], [], []
            for line in lines:
                value = [float(s) for s in line.split()]
                X.append(value[0])
                Y.append(value[1])
                Z.append(value[2]) if len(value) > 2 else Z.append(0.0)

        X = np.array(X)
        Y = np.array(Y)

        x_n = np.where(X == 0)
        x_id = x_n[0][0]
        size = len(X)

        return (X, Y, x_id, size)

    def datasplit(self, x, y, x_id, size):
        """
        Split the airfoil data into upper and lower surfaces.
        Note in the original .dat file the X values were in descending order for upper surface and ascending order for lower surface.
        While the corresponding Y values were in the opposite order of x
        so this function sorts the x values and reverse the y values to match the sorted x values.
        """

        x_up = sorted(x[0 : x_id + 1])  # sort the upper surface for x values
        y_up = y[0 : x_id + 1]
        x_up = np.array(x_up)
        y_up = np.array(y_up)
        y_up = y_up[::-1]  # reverse the y values to match the sorted x values

        x_low = sorted(x[x_id : size + 1])
        y_low = y[x_id : size + 1]
        x_low = np.array(x_low)
        y_low = np.array(y_low)

        return (x_up, y_up, x_low, y_low)

    def compute_half_alpha_thickness_te(self, x, y):
        """
        you wnat how  much the trailing edge is bent so we take the last point and
        3rd last point and find the difference to make a triangle to compute the angle using tangent = perp / base

        x: x positive direction data distribution
        y: y data distribution within x positive direction
        """
        x_n = len(x)
        x1 = x[x_n - 1]  # the last point
        y1 = y[x_n - 1]  # the last point
        x2 = x[x_n - 3]  # the third last point
        y2 = y[x_n - 3]  # the third last point

        deltax = abs(x1 - x2)
        deltay = abs(y1 - y2)
        # till here we've formed a right angle triangle to compute the angle

        half_alpha_te = mh.atan(deltay / deltax)  # in radian

        y_te = abs(y1)  # absolute value of the last y point

        return half_alpha_te, y_te

    def classfunction(self, x):
        """
        x: the x-direction data distribution of loaded airfoil data
        """
        N_1 = 0.5
        N_2 = 1.0
        x_n = len(x)
        C = np.empty((1, x_n))
        C = np.power(x, N_1) * np.power(1 - x, N_2)
        C = np.array(C)

        return C

    def bernstein(self, x):
        """
        Build the Bernstein polynomial basis matrix B for all x-points.

        - n = Bernstein order (example: n=5 ⇒ 6 basis functions)
        - x = array of x-coordinates along the airfoil surface

        Bernstein polynomial formula:

            B_j(x) = C(n, j) * x^j * (1 - x)^(n - j)

        where:

            C(n, j) = n! / ( j! * (n - j)! )

        The output B has shape:  [len(x), n+1]
        Row i = Bernstein basis evaluated at x[i].
        """

        n = self.Bernstein_order  # Bernstein polynomial order

        x_n = len(x)  # number of x-points
        B = np.empty((x_n, n + 1))  # matrix to store Bernstein values

        for i in range(0, x_n):  # iterate over x-locations
            for j in range(0, n + 1):  # iterate over polynomial index j (0..n)

                # Bernstein polynomial:
                # B[i, j] = C(n, j) * x[i]^j * (1 - x[i])^(n - j)
                B[i, j] = (
                    (mh.factorial(n) / (mh.factorial(j) * mh.factorial(n - j)))
                    * np.power(x[i], j)
                    * np.power(1 - x[i], n - j)
                )

        return B

    def shapefunction_fit(self, R_le, Y_te, alpha_te, x, y):
        """
        R_le: leading edge radius
        Y_te: half of tailing edge thickness
        alpha_te: half of tailing edge alpha (/rad)

        since y(x) = C(x) * S(x) => S(x) = (y(x) - x*Y_te) / C(x)

        """
        x_n = len(x)
        S = np.empty(
            (1, x_n)
        )  # make a matrix of dimension 1 x x_n to store the S values
        S[0] = np.sqrt(2 * R_le)  # leading edge condition i.e. S(0) = sqrt(2*R_le)
        S[0, x_n - 1] = (
            mh.tan(alpha_te) + Y_te
        )  # trailing edge condition i.e. S(1) = dy/dx at x=1 + Y_te

        C = self.classfunction(x)  # function calls

        for i in range(1, x_n - 1):
            S[0, i] = (y[i] - x[i] * Y_te) / C[i]

        return S

    # def comp_initial_control_points(self, R_le, Y_te, alpha_te, x, y):
    #     B = self.bernstein(x)  # function calls
    #     S = self.shapefunction_fit(R_le, Y_te, alpha_te, x, y) # function calls
    #     B_pinv = np.linalg.pinv(B)
    #     a = np.dot(B_pinv, S.T)
    #     a = np.array(a)

    #     return a

    def comp_initial_control_points(self, R_le, Y_te, alpha_te, x, y):
        """
        Compute initial control points 'a' for the shape function S(x) using least squares fitting.
        e.g. a would be [a0, a1, a2, ..., an] for n-order Bernstein polynomial
        YEH ABHI SMJH NHI AAYA!!
        """

        B = self.bernstein(
            x
        )  # function calls (why we calculating BERNSTEIN here again?)
        S = self.shapefunction_fit(R_le, Y_te, alpha_te, x, y)  # function calls
        B_pinv = np.linalg.pinv(B)  # find the pseudoinverse of B matrix
        a = np.dot(B_pinv, S.T)
        a = np.array(a)

        # --- ADD THIS PART TO PRINT THE WEIGHTS ---
        print("\n--- CST Weights (Control Points) ---")
        print(a.flatten())  # .flatten() prints it as a cleaner 1D list
        print("------------------------------------\n")
        # ------------------------------------------

        return a

    def shapefunction(self, a, B):
        """
        a: fitting control points
        B: n-order Bernstein polynomial
        """
        x_n = B.shape[0]
        a_n = B.shape[1]

        for i in range(0, x_n):
            for j in range(0, a_n):
                print(i, j)
                print("----------------")
                B[i, j] = B[i, j] * a[j]

        S = np.empty((x_n, 1))
        for i in range(0, x_n):
            S[i, 0] = B[i, :].sum()

        return S

    def CST_fitting(self, C, S, x, y_te):
        """
        C: class function
        S: shape function
        x: airfoil x-direction data distribution
        y_te: half of the tailing edge thickness
        """
        x_n = S.shape[0]
        y_CST = np.empty((x_n, 1))

        for i in range(0, x_n):
            y_CST[i, 0] = C[i] * S[i, 0] + x[i] * y_te

        return y_CST

    def error_eval(self, y, y_CST):
        error = y - y_CST

        return error

    def datsave(self, datname, x, y_fit):
        nu = len(x)
        with open(datname, "w") as f:
            for i in range(nu):
                f.write(str(x[i]) + "\t" + str(y_fit[i]) + "\t" + "0.0" + "\n")
        with open(datname, "r+") as f:
            content = f.read()
            f.seek(0, 0)
            f.write(str(nu) + "\n" + content)


# # testing
# def test_cst_basic():
#     """
#     Basic test function to verify CST functionality with simple test data
#     """
#     print("=== Testing CST Class ===")

#     # # Create simple test airfoil data (NACA-like shape)
#     # x_test = np.linspace(0, 1, 21)  # 21 points from 0 to 1
#     # y_upper = np.array([0.0, 0.05, 0.08, 0.09, 0.095, 0.09, 0.085, 0.08, 0.07, 0.06, 0.045,
#     #                    0.035, 0.025, 0.018, 0.012, 0.008, 0.005, 0.003, 0.002, 0.001, 0.0])
#     # y_lower = np.array([0.0, -0.02, -0.035, -0.04, -0.042, -0.04, -0.038, -0.035, -0.03, -0.025, -0.02,
#     #                    -0.015, -0.01, -0.008, -0.006, -0.004, -0.002, -0.001, -0.001, 0.0, 0.0])

#     # # Combine upper and lower surfaces (typical .dat file format)
#     # x_combined = np.concatenate([x_test[::-1], x_test[1:]])  # Upper surface reversed + lower surface
#     # y_combined = np.concatenate([y_upper[::-1], y_lower[1:]])

#     # Create temporary test file
#     test_filename = f"D:\\Anas\\Github\\FYP Organization Github\\optimization-team\\CST Maker\\Airfoils\\RAE2822.dat"


#     try:
#         # Test CST fitting
#         bernstein_order = 4
#         cst = CST(test_filename, bernstein_order)

#         print(f"Loading data from {test_filename}")
#         X, Y, x_id, size = cst.loaddata()
#         print(f"Loaded {size} points, leading edge index: {x_id}")

#         print("Splitting data into upper and lower surfaces...")
#         x_up, y_up, x_low, y_low = cst.datasplit(X, Y, x_id, size)
#         print(f"Upper surface: {len(x_up)} points, Lower surface: {len(x_low)} points")

#         # Test upper surface fitting
#         print("Testing upper surface fitting...")
#         R_le = 0.01  # Leading edge radius
#         alpha_te, Y_te = cst.compute_half_alpha_thickness_te(x_up, y_up)
#         print(f"Trailing edge angle: {alpha_te:.4f} rad, thickness: {Y_te:.6f}")

#         C_up = cst.classfunction(x_up)
#         B_up = cst.bernstein(x_up)
#         a_up = cst.comp_initial_control_points(R_le, Y_te, alpha_te, x_up, y_up)

#         print("CST fitting completed successfully!")
#         print(f"Upper surface control points shape: {a_up.shape}")

#     except Exception as e:
#         print(f"Test failed with error: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         # Clean up test file
#         import os
#         if os.path.exists(test_filename):
#             os.remove(test_filename)
#             print(f"Cleaned up {test_filename}")

# if __name__ == "__main__":
#     test_cst_basic()
