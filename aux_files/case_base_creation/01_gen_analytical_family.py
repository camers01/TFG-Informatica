import os
import pandas as pd

# 1. PATH SETUP 

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

INPUT_CSV = os.path.join(RAW_DATA_DIR, "xai_case_base.csv")
OUTPUT_CSV = os.path.join(PROCESSED_DATA_DIR, "checkpoint_case_base.csv")


# 2. THE MAPPING LOGIC

def determine_analytical_family(row):
    """
    Maps the combination of graph type, library, and scope 
    to the defined Analytical Family.
    """
    graph = row['xai_graph_type']
    library = row['library']
    scope = row['scope']

    # Family 1: Local_Attribution
    if scope == 'Local' and library == 'SHAP' and graph in ['Waterfall', 'Force', 'Bar']:
        return 'Local_Attribution'
    if scope == 'Local' and library == 'LIME' and graph in ['Bar', 'Dashboard']:
        return 'Local_Attribution'

    # Family 2: Global_Summary
    if scope == 'Global' and library == 'SHAP' and graph in ['Beeswarm', 'Violin', 'Bar']:
        return 'Global_Summary'

    # Family 3: Dependence_Curve
    if scope == 'Global' and library == 'SHAP' and graph == 'Scatter':
        return 'Dependence_Curve'
    if scope == 'Global' and library == 'ALE' and graph in ['1D', '2D']:
        return 'Dependence_Curve'

    # Family 4: Cohort_Pattern
    if scope == 'Cohort' and library == 'SHAP' and graph in ['Bar', 'Decision']:
        return 'Cohort_Pattern'

    # Family 5: Dense_Tracking
    if scope == 'Global' and library == 'SHAP' and graph == 'Heatmap':
        return 'Dense_Tracking'

    # Safety Fallback: Flags any edge cases not covered by the rules
    return 'Unknown'


# 3. EXECUTION

def main():

    print(f"Loading raw data from: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print("ERROR: Could not find xai_case_base.csv.")
        return

    print("Mapping analytical families...")
    df['analytical_family'] = df.apply(determine_analytical_family, axis=1) # Apply the function row by row

    # Print a quick summary to the terminal to verify the distribution
    print("\n--- Analytical Family Distribution ---")
    print(df['analytical_family'].value_counts())
    print("--------------------------------------\n")

    # Flag if there are any unknowns
    if 'Unknown' in df['analytical_family'].values:
        print("WARNING: Some cases were marked as 'Unknown'.")

    # Dynamic column reordering to ensure 'solution_insights' is at the end, regardless of its original position
    cols = df.columns.tolist()
    if 'solution_insights' in cols:
        cols.remove('solution_insights')
        cols.append('solution_insights') # Appends it to the end
    df = df[cols] # Reassign the dataframe to the new order

    print(f"Saving to processed folder: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Process Completed.")

if __name__ == "__main__":
    main()