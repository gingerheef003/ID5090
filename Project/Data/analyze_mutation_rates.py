import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np # For calculations

# --- Configuration ---
# Define the mutation rates and corresponding filenames
# Adjust filenames if they differ slightly (e.g., underscores vs spaces)
file_mapping = {
    0.1:    "Project/Data/evolution_data 0_1.csv",
    0.01:   "Project/Data/evolution_data 0_01.csv",
    0.001:  "Project/Data/evolution_data 0_001.csv",
    0.0001: "Project/Data/evolution_data 0_0001.csv",
}

# Define the generation where the environment changes (safe zone switch)
# Adjust this based on your MAX_GENERATIONS setting in the simulation
# Example: If MAX_GENERATIONS = 200, the switch happens after generation 99 (at start of gen 100)
SWITCH_GENERATION = 100 # Generation number *after* which the switch occurs

# Threshold for recovery calculation (e.g., % survival rate)
RECOVERY_THRESHOLD = 50.0

# --- Load Data ---
dataframes = {}
mutation_rates = sorted(file_mapping.keys()) # Load in sorted order

print("Loading data...")
for rate in mutation_rates:
    filename = file_mapping[rate]
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            # Ensure columns have correct types
            df['Generation'] = df['Generation'].astype(int)
            df['SurvivalRate'] = pd.to_numeric(df['SurvivalRate'], errors='coerce')
            df['GeneticDiversity'] = pd.to_numeric(df['GeneticDiversity'], errors='coerce')
            df = df.dropna(subset=['SurvivalRate', 'GeneticDiversity']) # Drop rows with loading errors
            dataframes[rate] = df
            print(f"  Loaded data for mutation rate {rate} from {filename} ({len(df)} generations)")
        except Exception as e:
            print(f"  Error loading or processing {filename}: {e}")
    else:
        print(f"  Warning: File not found - {filename}")

if not dataframes:
    print("Error: No data loaded. Exiting.")
    exit()

# --- Plotting ---
print("\nGenerating plots...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Impact of Mutation Rate on Evolution Dynamics', fontsize=16)

# Plot Survival Rate
ax1.set_title('Survival Rate over Generations')
ax1.set_ylabel('Survival Rate (%)')
ax1.grid(True, linestyle='--', alpha=0.6)
# Add vertical line for environment switch
ax1.axvline(x=SWITCH_GENERATION, color='r', linestyle='--', label=f'Env. Switch (Gen {SWITCH_GENERATION})')


# Plot Genetic Diversity
ax2.set_title('Genetic Diversity over Generations')
ax2.set_xlabel('Generation')
ax2.set_ylabel('Unique Gene Combinations')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.axvline(x=SWITCH_GENERATION, color='r', linestyle='--') # No label needed here again


# Plot data for each mutation rate
for rate, df in dataframes.items():
    ax1.plot(df['Generation'], df['SurvivalRate'], marker='.', markersize=4, linestyle='-', label=f'Rate = {rate}')
    ax2.plot(df['Generation'], df['GeneticDiversity'], marker='.', markersize=4, linestyle='-', label=f'Rate = {rate}')

# Add legends
ax1.legend(loc='best')
ax2.legend(loc='best')

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.savefig("mutation_rate_comparison_plots.png")
print("  Saved plots to mutation_rate_comparison_plots.png")
# plt.show() # Uncomment to display plots immediately

# --- Quantitative Analysis ---
print("\nPerforming quantitative analysis...")
analysis_results = []

for rate, df in dataframes.items():
    if df.empty:
        continue

    # Phase 1: Before the switch (up to and including SWITCH_GENERATION - 1)
    df_phase1 = df[df['Generation'] < SWITCH_GENERATION]
    avg_survival_p1 = df_phase1['SurvivalRate'].mean() if not df_phase1.empty else np.nan
    avg_diversity_p1 = df_phase1['GeneticDiversity'].mean() if not df_phase1.empty else np.nan

    # Phase 2: After the switch (from SWITCH_GENERATION onwards)
    df_phase2 = df[df['Generation'] >= SWITCH_GENERATION]
    avg_survival_p2 = df_phase2['SurvivalRate'].mean() if not df_phase2.empty else np.nan
    avg_diversity_p2 = df_phase2['GeneticDiversity'].mean() if not df_phase2.empty else np.nan

    # Time to recover (generations after switch to reach threshold)
    recovery_gen = np.nan
    if not df_phase2.empty:
        recovered_df = df_phase2[df_phase2['SurvivalRate'] >= RECOVERY_THRESHOLD]
        if not recovered_df.empty:
            first_recovery_gen = recovered_df['Generation'].min()
            recovery_gen = first_recovery_gen - SWITCH_GENERATION

    analysis_results.append({
        'Mutation Rate': rate,
        'Avg Survival (Phase 1)': avg_survival_p1,
        'Avg Diversity (Phase 1)': avg_diversity_p1,
        'Avg Survival (Phase 2)': avg_survival_p2,
        'Avg Diversity (Phase 2)': avg_diversity_p2,
        f'Generations to Recover (>{RECOVERY_THRESHOLD}%)': recovery_gen
    })

# Display results in a table
if analysis_results:
    results_df = pd.DataFrame(analysis_results)
    print("\nQuantitative Summary:")
    # Format floats for better readability
    pd.options.display.float_format = '{:.2f}'.format
    print(results_df.to_string(index=False)) # Use to_string for better console formatting
else:
    print("\nNo results to display for quantitative analysis.")

print("\nAnalysis complete.")