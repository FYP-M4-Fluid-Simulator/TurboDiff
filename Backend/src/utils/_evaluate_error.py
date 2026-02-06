import numpy as np


# --- UPDATED FUNCTION ---
def measure_closeness(x_actual, y_actual, x_pred, y_pred):
    """
    1. Calculates the explicit difference (error) for X and Y separately.
    2. Calculates a combined geometric accuracy percentage.
    """
    # Flatten arrays to ensure 1D
    x_act = np.array(x_actual).flatten()
    y_act = np.array(y_actual).flatten()
    x_pre = np.array(x_pred).flatten()
    y_pre = np.array(y_pred).flatten()



    # --- 1. Calculate Differences (Absolute Errors) ---
    diff_x = np.abs(x_act - x_pre)
    diff_y = np.abs(y_act - y_pre)
    
    avg_diff_x = np.mean(diff_x)
    avg_diff_y = np.mean(diff_y)
    max_diff_y = np.max(diff_y)

    # --- 2. Calculate Percentage Accuracy (Euclidean Method) ---
    # Distance between actual and predicted points
    dist_error = np.sqrt(diff_x**2 + diff_y**2)
    
    # Magnitude of actual points (distance from origin)
    magnitude = np.sqrt(x_act**2 + y_act**2)
    
    # Handle division by zero (at 0,0)
    magnitude[magnitude == 0] = 1e-10
    
    # Relative error per point
    relative_error = dist_error / magnitude
    
    # Overall accuracy
    accuracy_pct = (1 - np.mean(relative_error)) * 100

    # --- 3. Print Report ---
    print("\n" + "="*40)
    print("       FITTING ACCURACY REPORT       ")
    print("="*40)
    print(f"Average Difference in X : {avg_diff_x:.6f}")
    print(f"Average Difference in Y : {avg_diff_y:.6f}")
    print(f"Max Deviation in Y      : {max_diff_y:.6f}")
    print("-" * 40)
    print(f"Overall Model Fidelity  : {accuracy_pct:.4f}%")
    print("="*40 + "\n")
    
    return accuracy_pct
# ------------------------