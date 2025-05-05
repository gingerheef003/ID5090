import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

import pysindy as ps

# --- Configuration ---
file_mapping = {
    0.1:    "Project/Data/evolution_data 0_1.csv",
    0.01:   "Project/Data/evolution_data 0_01.csv",
    0.001:  "Project/Data/evolution_data 0_001.csv",
    0.0001: "Project/Data/evolution_data 0_0001.csv",
}
SWITCH_GENERATION = 100
RECOVERY_THRESHOLD = 50.0

# --- SINDy Configuration ---
SINDY_POLY_ORDER = 2      # Max degree for polynomial terms in the library
SINDY_THRESHOLD = 0.05    # Sparsity threshold for STLSQ optimizer
SINDY_INCLUDE_INTERACTION = True # Include terms like S*D

# --- Load Data ---
dataframes = {}
mutation_rates = sorted(file_mapping.keys())

print("Loading data...")
# ...(same loading logic as before)...
for rate in mutation_rates:
    filename = file_mapping[rate]
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            df['Generation'] = df['Generation'].astype(int)
            # Normalize SurvivalRate to be between 0 and 1 for potentially better SINDy stability
            df['SurvivalRate'] = pd.to_numeric(df['SurvivalRate'], errors='coerce') / 100.0
            df['GeneticDiversity'] = pd.to_numeric(df['GeneticDiversity'], errors='coerce')
            # Optional: Smooth data if it's very noisy (e.g., rolling average)
            # df['SurvivalRate'] = df['SurvivalRate'].rolling(window=5, center=True, min_periods=1).mean()
            # df['GeneticDiversity'] = df['GeneticDiversity'].rolling(window=5, center=True, min_periods=1).mean()
            df = df.dropna(subset=['SurvivalRate', 'GeneticDiversity'])
            if len(df) > 5: # Need minimum points for SINDy
                dataframes[rate] = df
                print(f"  Loaded data for mutation rate {rate} from {filename} ({len(df)} generations)")
            else:
                 print(f"  Skipping rate {rate} due to insufficient data points after cleaning ({len(df)}).")
        except Exception as e:
            print(f"  Error loading or processing {filename}: {e}")
    else:
        print(f"  Warning: File not found - {filename}")

if not dataframes:
    print("Error: No data loaded. Exiting.")
    exit()


# --- Plotting ---
print("\nGenerating comparison plots...")
# ...(same plotting logic as before, but adjust Survival Rate axis if normalized)...
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Impact of Mutation Rate on Evolution Dynamics', fontsize=16)
ax1.set_title('Survival Rate over Generations')
ax1.set_ylabel('Survival Rate (Fraction)') # Adjusted label
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.axvline(x=SWITCH_GENERATION, color='r', linestyle='--', label=f'Env. Switch (Gen {SWITCH_GENERATION})')
ax2.set_title('Genetic Diversity over Generations')
ax2.set_xlabel('Generation')
ax2.set_ylabel('Unique Gene Combinations')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.axvline(x=SWITCH_GENERATION, color='r', linestyle='--')

for rate, df in dataframes.items():
    ax1.plot(df['Generation'], df['SurvivalRate'], marker='.', markersize=4, linestyle='-', label=f'Rate = {rate}')
    ax2.plot(df['Generation'], df['GeneticDiversity'], marker='.', markersize=4, linestyle='-', label=f'Rate = {rate}')

ax1.legend(loc='best')
ax2.legend(loc='best')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("mutation_rate_comparison_plots.png")
print("  Saved plots to mutation_rate_comparison_plots.png")
# plt.show()

# --- Quantitative Analysis ---
print("\nPerforming quantitative analysis...")
# ...(same quantitative analysis logic as before)...
# Remember SurvivalRate is now 0-1, adjust threshold if needed (e.g., 0.5 instead of 50.0)
RECOVERY_THRESHOLD_NORMALIZED = RECOVERY_THRESHOLD / 100.0
analysis_results = []
for rate, df in dataframes.items():
    # ... calculation logic ...
    # Use RECOVERY_THRESHOLD_NORMALIZED for comparison
    recovered_df = df_phase2[df_phase2['SurvivalRate'] >= RECOVERY_THRESHOLD_NORMALIZED]
    # ... rest of calculations ...
    analysis_results.append({
        'Mutation Rate': rate,
        'Avg Survival (Phase 1)': avg_survival_p1 * 100, # Convert back for display
        'Avg Diversity (Phase 1)': avg_diversity_p1,
        'Avg Survival (Phase 2)': avg_survival_p2 * 100, # Convert back for display
        'Avg Diversity (Phase 2)': avg_diversity_p2,
        f'Generations to Recover (>{RECOVERY_THRESHOLD}%)': recovery_gen
    })
if analysis_results:
    results_df = pd.DataFrame(analysis_results)
    print("\nQuantitative Summary:")
    pd.options.display.float_format = '{:.2f}'.format
    print(results_df.to_string(index=False))
else:
    print("\nNo results to display for quantitative analysis.")


# --- SINDy Analysis for Population Dynamics ---
print("\n--- Running SINDy Analysis for Population Dynamics ---")


sindy_models = {}
plt.figure(figsize=(12, 6 * len(dataframes))) # Create figure for SINDy results

plot_index = 1
for rate, df in dataframes.items():
    print(f"\nAnalyzing Mutation Rate: {rate}")

    # Prepare data for SINDy
    # Ensure data is sorted by Generation
    df = df.sort_values(by='Generation')
    t = df['Generation'].values
    X = df[['SurvivalRate', 'GeneticDiversity']].values

    if len(t) < SINDY_POLY_ORDER + 2: # Need enough points for differentiation and fitting
          print(f"  Skipping SINDy for rate {rate}: Insufficient data points ({len(t)})")
          continue

    # Define SINDy model components
    # Use Finite Differences for differentiation
    diff_method = ps.FiniteDifference(order=2)
    # Define feature library (Polynomials of Survival Rate 'S' and Diversity 'D')
    feature_library = ps.PolynomialLibrary(degree=SINDY_POLY_ORDER, include_interaction=SINDY_INCLUDE_INTERACTION)
    # Define optimizer
    optimizer = ps.STLSQ(threshold=SINDY_THRESHOLD, normalize_columns=True)
    # Feature names for easier equation reading
    feature_names = ['S', 'D']

    # Initialize SINDy model
    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=feature_library,
        differentiation_method=diff_method,
        feature_names=feature_names
    )

    try:
        # Fit the model to the data
        model.fit(X, t=t)

        # Print the discovered equations
        print(f"  Discovered Equations (Rate = {rate}):")
        model.print(lhs=['dS/dt', 'dD/dt']) # d/dt here means d/dGeneration
        sindy_models[rate] = model # Store the fitted model

        # --- Plot SINDy Simulation vs Original Data ---
        ax = plt.subplot(len(dataframes), 2, plot_index)
        # Simulate model from initial condition
        t_sim = np.linspace(t[0], t[-1], num=len(t)*2) # Finer time steps for simulation
        try:
              X_sim = model.simulate(X[0], t=t_sim) # Start simulation from the first data point
              
              # Plot Survival Rate (S)
              ax.plot(t, X[:, 0], 'bo', label='Original Survival Rate', markersize=4, alpha=0.6)
              ax.plot(t_sim, X_sim[:, 0], 'b-', label='SINDy Model (S)')
              ax.set_ylabel('Survival Rate (S)')

        except Exception as sim_error:
              print(f"  Warning: SINDy simulation failed for S: {sim_error}")
              ax.plot(t, X[:, 0], 'bo', label='Original Survival Rate', markersize=4, alpha=0.6)


        ax.set_title(f'SINDy Fit (Rate={rate}) - Survival Rate')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        plot_index += 1

        ax = plt.subplot(len(dataframes), 2, plot_index)
        try:
              # X_sim already calculated from above
              # Plot Genetic Diversity (D)
              ax.plot(t, X[:, 1], 'go', label='Original Genetic Diversity', markersize=4, alpha=0.6)
              ax.plot(t_sim, X_sim[:, 1], 'g-', label='SINDy Model (D)')
              ax.set_ylabel('Genetic Diversity (D)')
        except Exception as sim_error:
              print(f"  Warning: SINDy simulation failed for D: {sim_error}")
              ax.plot(t, X[:, 1], 'go', label='Original Genetic Diversity', markersize=4, alpha=0.6)


        ax.set_title(f'SINDy Fit (Rate={rate}) - Genetic Diversity')
        ax.set_xlabel('Generation')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        plot_index += 1

    except Exception as e:
        print(f"  Error during SINDy fitting or plotting for rate {rate}: {e}")
        # Advance plot index even if error occurs to maintain layout
        if plot_index % 2 != 0: # If error happened on first plot
              plot_index += 2
        else: # If error happened on second plot
              plot_index += 1


plt.tight_layout()
plt.savefig("mutation_rate_sindy_fits.png")
print("\n  Saved SINDy fit plots to mutation_rate_sindy_fits.png")

print("\nAnalysis complete.")