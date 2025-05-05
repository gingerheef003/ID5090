# analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast  # For parsing string representation of DNA list/tuples
import os

# Import SINDy-related libraries (install if necessary: pip install pysindy)
try:
    import pysindy as ps
    PYSINDY_AVAILABLE = True
except ImportError:
    print("Warning: PySINDy not found. SINDy analysis will be unavailable.")
    print("Install using: pip install pysindy")
    PYSINDY_AVAILABLE = False

# Import Scikit-learn for Lasso and preprocessing
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.exceptions import ConvergenceWarning
import warnings

# Ignore convergence warnings from Lasso for cleaner output
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# --- Configuration ---
DATA_DIR = "simulation_output" # Directory where simulation saved data
SINDY_DATA_FILE = os.path.join(DATA_DIR, "final_sim_data_sindy.csv")
LASSO_DATA_FILE = os.path.join(DATA_DIR, "final_sim_data_lasso.csv")

# --- Helper Functions ---

def load_data(filepath):
    """Loads data from CSV, handling potential file not found errors."""
    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}")
        return None
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None

def parse_dna(dna_str):
    """Safely parses the DNA string back into a list of tuples."""
    try:
        # Split the string by semicolon and then use ast.literal_eval for each part
        # This assumes the tuples were saved like "(a, b, c)"
        # Adjust parsing if the format is different
        genes = [ast.literal_eval(gene) for gene in dna_str.split(';') if gene]
        return genes
    except Exception as e:
        # print(f"Warning: Could not parse DNA string: {dna_str}. Error: {e}")
        return [] # Return empty list on failure

# --- SINDy Analysis ---

def analyze_movement_sindy(sindy_df):
    """
    Applies SINDy to discover movement equations from creature trajectory data.
    """
    if not PYSINDY_AVAILABLE:
        print("SINDy analysis skipped (PySINDy not available).")
        return

    if sindy_df is None or sindy_df.empty:
        print("SINDy analysis skipped (no data loaded).")
        return

    print("\n--- Running SINDy Analysis for Movement ---")

    # Configuration for SINDy
    POLY_ORDER = 2
    THRESHOLD = 0.05 # Sparsity threshold

    # Calculate dt (time difference between consecutive points for *each* creature)
    # Need to be careful with grouping by creature_id and sorting by time
    sindy_df = sindy_df.sort_values(by=['creature_id', 'time'])
    sindy_df['dt'] = sindy_df.groupby('creature_id')['time'].diff().fillna(0)

    # Calculate velocities (dx/dt, dy/dt) using positions *before* and *after* the step
    # Avoid division by zero if dt is 0 (first point or error)
    sindy_df['dx_dt'] = np.where(sindy_df['dt'] > 0, (sindy_df['x_after'] - sindy_df['x_before']) / sindy_df['dt'], 0)
    sindy_df['dy_dt'] = np.where(sindy_df['dt'] > 0, (sindy_df['y_after'] - sindy_df['y_before']) / sindy_df['dt'], 0)

    # Identify sensor columns (heuristic: columns not time, id, generation, pos, dt, velocity)
    exclude_cols = ['time', 'creature_id', 'generation', 'x_before', 'y_before',
                    'x_after', 'y_after', 'dt', 'dx_dt', 'dy_dt']
    sensor_cols = [col for col in sindy_df.columns if col not in exclude_cols]

    if not sensor_cols:
        print("Error: No sensor columns found for SINDy analysis.")
        return

    print(f"Using sensors for SINDy: {sensor_cols}")

    # Prepare data for SINDy: X contains state variables (sensors), X_dot contains derivatives
    # We use the state *before* the move (x_before, y_before, corresponding sensors)
    # to predict the velocity (dx_dt, dy_dt) calculated from the move.
    X = sindy_df[sensor_cols].values
    X_dot = sindy_df[['dx_dt', 'dy_dt']].values
    t = sindy_df['time'].values

    # --- Fit SINDy Model ---
    # Define feature library (e.g., polynomial features of sensors)
    feature_library = ps.PolynomialLibrary(degree=POLY_ORDER)
    # You could add other libraries like ps.FourierLibrary if oscillators are important

    # Define optimizer (Sparse Thresholded Least Squares)
    optimizer = ps.STLSQ(threshold=THRESHOLD, normalize_columns=True)

    # Initialize and fit SINDy model
    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=feature_library,
        feature_names=sensor_cols # Pass sensor names for interpretable equations
    )

    try:
        # Filter out rows where dt was 0 or very small? Or handle potential NaNs/Infs in X, X_dot
        valid_rows = sindy_df['dt'] > 1e-6 # Example filter
        if np.sum(valid_rows) < 10: # Need sufficient data points
             print("Warning: Not enough valid data points for SINDy fitting after filtering.")
             return
             
        model.fit(X[valid_rows], x_dot=X_dot[valid_rows], t=t[valid_rows]) # Provide time if needed by library/differentiation

        print("\nDiscovered Movement Equations (dx/dt, dy/dt):")
        # Print equations for dx/dt (index 0) and dy/dt (index 1)
        model.print(lhs=['dx/dt', 'dy/dt'])

        # --- Optional: Simulate discovered model (more advanced) ---
        # You could try simulating the discovered ODEs and comparing trajectories

    except Exception as e:
        print(f"Error during SINDy fitting: {e}")
        import traceback
        traceback.print_exc()


# --- Lasso Analysis ---

def analyze_survival_lasso_sensors(lasso_df):
    """
    Uses Lasso regression to identify which average sensor readings predict survival.
    """
    if lasso_df is None or lasso_df.empty:
        print("Lasso analysis (sensors) skipped (no data loaded).")
        return

    print("\n--- Running Lasso Analysis for Survival (Sensors) ---")

    # Identify average sensor columns
    sensor_avg_cols = [col for col in lasso_df.columns if col.startswith('avg_')]
    if not sensor_avg_cols:
        print("Error: No average sensor columns found for Lasso analysis.")
        return

    print(f"Using average sensor features: {sensor_avg_cols}")

    # Prepare data
    X = lasso_df[sensor_avg_cols].fillna(0) # Fill missing sensor averages with 0
    y = lasso_df['survived']

    if len(X) < 10:
        print("Warning: Not enough data points for reliable Lasso analysis.")
        return
        
    # Split data (optional but good practice)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Fit Lasso Model ---
    # Use LassoCV to find the best regularization strength (alpha)
    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=5000, n_jobs=-1)
    lasso_cv.fit(X_train_scaled, y_train)

    print(f"Best alpha found by LassoCV: {lasso_cv.alpha_:.4f}")

    # Evaluate on test set
    y_pred = lasso_cv.predict(X_test_scaled)
    # Since Lasso outputs continuous values, threshold for classification metric
    y_pred_class = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_class)
    print(f"\nLasso Test Set Accuracy (thresholded at 0.5): {accuracy:.3f}")
    # print("Classification Report:")
    # print(classification_report(y_test, y_pred_class))

    # --- Analyze Coefficients ---
    coefficients = pd.Series(lasso_cv.coef_, index=sensor_avg_cols)
    important_coeffs = coefficients[coefficients.abs() > 1e-4] # Filter near-zero coeffs

    if important_coeffs.empty:
        print("\nLasso selected no significant sensor features.")
    else:
        print("\nLasso Coefficients (Importance of Avg Sensors for Survival):")
        # Sort by absolute value for importance ranking
        print(important_coeffs.abs().sort_values(ascending=False))

        # Plotting coefficients
        plt.figure(figsize=(10, max(6, len(important_coeffs) * 0.4)))
        important_coeffs.sort_values().plot(kind='barh')
        plt.title('Lasso Coefficients for Survival Prediction (Sensors)')
        plt.xlabel('Coefficient Value (Importance)')
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, "lasso_sensor_importance.png"))
        print(f"Saved sensor importance plot to {os.path.join(DATA_DIR, 'lasso_sensor_importance.png')}")
        # plt.show() # Optional: Show plot immediately

def analyze_survival_lasso_genes(lasso_df):
    """
    Uses Lasso regression to identify which genes or gene components predict survival.
    (Requires careful feature engineering from DNA string).
    """
    if lasso_df is None or lasso_df.empty:
        print("Lasso analysis (genes) skipped (no data loaded).")
        return

    print("\n--- Running Lasso Analysis for Survival (Genes) ---")
    print("Note: Gene analysis requires robust DNA parsing and feature engineering.")

    # --- Feature Engineering (Example: Count Sensor/Action types) ---
    # This is a simplified approach. A better one might create binary columns
    # for specific gene connections (e.g., 'IH_Sfd_H0', 'HO_H3_Mfd').

    all_sensors = set(POSSIBLE_SENSORS) # Need POSSIBLE_SENSORS from simulation
    all_actions = set(POSSIBLE_ACTIONS) # Need POSSIBLE_ACTIONS from simulation

    feature_data = []
    for index, row in lasso_df.iterrows():
        dna = parse_dna(row['dna'])
        features = {}
        # Count occurrences of each sensor type in IH genes
        sensor_counts = defaultdict(int)
        action_counts = defaultdict(int)
        ih_count = 0
        ho_count = 0
        for gene in dna:
            try:
                if gene[0] == 'IH' and gene[1] in all_sensors:
                    sensor_counts[gene[1]] += 1
                    ih_count += 1
                elif gene[0] == 'HO' and gene[2] in all_actions:
                    action_counts[gene[2]] += 1
                    ho_count += 1
            except (IndexError, TypeError): # Handle malformed gene tuples
                continue # Skip this gene if parsing failed inside loop

        # Add features like 'count_sensor_Sfd', 'count_action_Mfd'
        for s in all_sensors: features[f'count_sensor_{s}'] = sensor_counts[s]
        for a in all_actions: features[f'count_action_{a}'] = action_counts[a]
        features['num_IH_genes'] = ih_count
        features['num_HO_genes'] = ho_count
        feature_data.append(features)

    X_gene_features = pd.DataFrame(feature_data).fillna(0)
    y = lasso_df['survived']

    if X_gene_features.empty:
        print("Error: No gene features could be extracted.")
        return
        
    if len(X_gene_features) < 10:
        print("Warning: Not enough data points for reliable Lasso analysis.")
        return

    print(f"Using gene features (counts): {list(X_gene_features.columns)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_gene_features, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit LassoCV
    lasso_cv_genes = LassoCV(cv=5, random_state=42, max_iter=5000, n_jobs=-1)
    lasso_cv_genes.fit(X_train_scaled, y_train)

    print(f"Best alpha found by LassoCV (genes): {lasso_cv_genes.alpha_:.4f}")

    # Evaluate
    y_pred_class = (lasso_cv_genes.predict(X_test_scaled) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_class)
    print(f"Lasso Test Set Accuracy (genes, thresholded): {accuracy:.3f}")

    # Analyze Coefficients
    coefficients_genes = pd.Series(lasso_cv_genes.coef_, index=X_gene_features.columns)
    important_coeffs_genes = coefficients_genes[coefficients_genes.abs() > 1e-4]

    if important_coeffs_genes.empty:
        print("\nLasso selected no significant gene features (using counts).")
    else:
        print("\nLasso Coefficients (Importance of Gene Features for Survival):")
        print(important_coeffs_genes.abs().sort_values(ascending=False))

        plt.figure(figsize=(10, max(6, len(important_coeffs_genes) * 0.4)))
        important_coeffs_genes.sort_values().plot(kind='barh')
        plt.title('Lasso Coefficients for Survival Prediction (Gene Counts)')
        plt.xlabel('Coefficient Value (Importance)')
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, "lasso_gene_importance.png"))
        print(f"Saved gene importance plot to {os.path.join(DATA_DIR, 'lasso_gene_importance.png')}")
        # plt.show() # Optional


# --- Main Execution ---
if __name__ == "__main__":
    print("Loading simulation data...")
    sindy_df = load_data(SINDY_DATA_FILE)
    lasso_df = load_data(LASSO_DATA_FILE)

    # Run Analyses
    analyze_movement_sindy(sindy_df)
    analyze_survival_lasso_sensors(lasso_df)
    analyze_survival_lasso_genes(lasso_df) # Requires POSSIBLE_SENSORS/ACTIONS

    print("\nAnalysis complete.")
    # Keep plots open if shown interactively
    if any(plt.get_fignums()):
         print("Close plot windows to exit.")
         plt.show()