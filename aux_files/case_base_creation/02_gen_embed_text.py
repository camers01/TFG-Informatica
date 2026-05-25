import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. PATH SETUP

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# We read from and write to the same checkpoint file from Script 1
CHECKPOINT_CSV = os.path.join(PROCESSED_DATA_DIR, "checkpoint_case_base.csv")

# Directories for the new embeddings
DOMAIN_EMB_DIR = os.path.join(PROCESSED_DATA_DIR, "embeddings_domain")
TASK_EMB_DIR = os.path.join(PROCESSED_DATA_DIR, "embeddings_task")


# 2. EXECUTION

def main():

    print(f"Loading checkpoint data from: {CHECKPOINT_CSV}")
    try:
        # Enforce 'id' as a string
        df = pd.read_csv(CHECKPOINT_CSV, dtype={'id': str}) 
    except FileNotFoundError:
        print("ERROR: Checkpoint CSV not found.")
        return

    print("Loading the MiniLM model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Extract text as lists
    domain_texts = df['domain'].astype(str).tolist()
    task_texts = df['ai_task'].astype(str).tolist()

    print(f"Computing embeddings for {len(domain_texts)} domains...")
    domain_embeddings = model.encode(domain_texts, show_progress_bar=True, normalize_embeddings=True) # Batch encoding is significantly faster than looping row by row

    print(f"Computing embeddings for {len(task_texts)} tasks...")
    task_embeddings = model.encode(task_texts, show_progress_bar=True, normalize_embeddings=True) # Batch encoding is significantly faster than looping row by row

    print("Saving .npy files and updating dataframe...")
    domain_paths = []
    task_paths = []

    for i, row in df.iterrows():

        # Ensure the ID is exactly 6 digits (e.g., '000001')
        case_id = str(row['id']).zfill(6) 

        # Define absolute paths for saving the files
        domain_abs_path = os.path.join(DOMAIN_EMB_DIR, f"{case_id}.npy")
        task_abs_path = os.path.join(TASK_EMB_DIR, f"{case_id}.npy")

        # Save the numpy arrays to the respective directories
        np.save(domain_abs_path, domain_embeddings[i])
        np.save(task_abs_path, task_embeddings[i])

        # Store the clean, relative paths for the CSV
        domain_paths.append(f"embeddings_domain/{case_id}.npy")
        task_paths.append(f"embeddings_task/{case_id}.npy")

    # Add the new path columns to the dataframe
    df['domain_emb_path'] = domain_paths
    df['task_emb_path'] = task_paths

    # Dynamic column reordering
    all_cols = df.columns.tolist() # Grab all current columns
    path_cols = ['domain_emb_path', 'task_emb_path'] # We want to group together the paths right after 'img_path'
    metadata_cols = [c for c in all_cols if c not in ['id', 'img_path', 'solution_insights'] + path_cols] # Isolate the remaining metadata columns
    final_order = ['id', 'img_path'] + path_cols + metadata_cols # Reconstruct the list in the perfect order
    if 'solution_insights' in all_cols: # Ensure solution_insights is always last
        final_order.append('solution_insights')
    df = df[final_order]

    print(f"Saving updated data to: {CHECKPOINT_CSV}")
    df.to_csv(CHECKPOINT_CSV, index=False)
    print("Process Completed.")

if __name__ == "__main__":
    main()