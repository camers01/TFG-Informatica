import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F
from tqdm import tqdm


# 1. PATH SETUP

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Read from and write to our dynamic checkpoint file
CHECKPOINT_CSV = os.path.join(PROCESSED_DATA_DIR, "checkpoint_case_base.csv")

# Directory for visual embeddings
VISUAL_EMB_DIR = os.path.join(PROCESSED_DATA_DIR, "embeddings_visual")


# 2. CONFIGURATION

BATCH_SIZE = 4
MODEL_NAME = "ahmed-masry/chartgemma"

# Use GPU if available, otherwise fallback to CPU safely
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 3. EXECUTION

def main():

    print(f"Loading checkpoint data from: {CHECKPOINT_CSV}")
    try:
        df = pd.read_csv(CHECKPOINT_CSV, dtype={'id': str})
    except FileNotFoundError:
        print("ERROR: Checkpoint CSV not found. Run previous scripts first.")
        return

    # Ensure the column exists so we can check for missing values
    if 'visual_emb_path' not in df.columns:
        df['visual_emb_path'] = pd.NA

    # Find only the rows that haven't been processed yet
    missing_mask = df['visual_emb_path'].isna()
    rows_to_process = df[missing_mask]

    if rows_to_process.empty:
        print("All visual embeddings are already computed.")
        return

    print(f"Found {len(rows_to_process)} images left to process.")
    print(f"Loading Vision Model onto {DEVICE.upper()}...")
    
    # Load the specific PaliGemma architecture for ChartGemma
    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    # Process in chunks to prevent memory leaks
    for start_idx in tqdm(range(0, len(rows_to_process), BATCH_SIZE), desc="Processing Image Batches"):
        
        batch_df = rows_to_process.iloc[start_idx:start_idx + BATCH_SIZE]
        images = []
        valid_indices = []

        # 1. Safely load images
        for idx, row in batch_df.iterrows():
            img_abs_path = os.path.join(RAW_DATA_DIR, row['img_path'])
            try:
                img = Image.open(img_abs_path).convert("RGB")
                images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                print(f"\nWARNING: Could not load image {img_abs_path}. Error: {e}")
                continue
        
        if not images:
            continue

        # 2. Process through the Vision Tower
        with torch.no_grad():

            # Prepare pixels (PaliGemma format) - only using image_processor (no text inputs here)
            inputs = processor.image_processor(images=images, return_tensors="pt").to(DEVICE)
            
            # Extract the vector from the vision tower (inputs directly contains 'pixel_values', instead of inputs.pixel_values))
            vision_outputs = model.vision_tower(inputs["pixel_values"])
            
            # Take the mean across the sequence dimension (dim=1) 
            raw_embeddings = vision_outputs.last_hidden_state.mean(dim=1)

            # L2 Normalization to save computation time in Phase 2
            normalized_embeddings = F.normalize(raw_embeddings, p=2, dim=1)

            # Move to CPU and convert to Numpy for saving
            pooled_embeddings = normalized_embeddings.cpu().numpy()

        # 3. Save and update DataFrame
        for i, df_idx in enumerate(valid_indices):
            case_id = str(df.at[df_idx, 'id']).zfill(6)
            save_path = os.path.join(VISUAL_EMB_DIR, f"{case_id}.npy")
            
            np.save(save_path, pooled_embeddings[i])
            
            # Store relative path in the main DataFrame
            df.at[df_idx, 'visual_emb_path'] = f"embeddings_visual/{case_id}.npy"

        # 4. Dynamic Column Reordering (Paths at the front, Insights at the back)
        all_cols = df.columns.tolist()
        path_cols = ['domain_emb_path', 'task_emb_path', 'visual_emb_path']
        metadata_cols = [c for c in all_cols if c not in ['id', 'img_path', 'solution_insights'] + path_cols] # Isolate metadata (excluding paths, id, img_path, and solution_insights)
        final_order = ['id', 'img_path'] + path_cols + metadata_cols # Reconstruct the order
        if 'solution_insights' in all_cols:
            final_order.append('solution_insights')
        df = df[final_order]

        # 5. Save Checkpoint after EVERY batch
        df.to_csv(CHECKPOINT_CSV, index=False)
        
        # Clear GPU cache to prevent out-of-memory errors on long runs
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("Process Completed.")

if __name__ == "__main__":
    main()