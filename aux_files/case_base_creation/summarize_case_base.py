import os
import pandas as pd
from tqdm import tqdm
import gc
import torch

from hallucination_detection.manager import HallucinationManager
from hallucination_detection.cross_modal.jina_clip import JinaClipEvaluator
from hallucination_detection.cross_output.eigen_score import EigenScoreEvaluator

from summarizer import SynthesizerManager


# 1. PATH SETUP 

# Access the root directory by going one level up from 'src'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define sub-directories
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Define input and output files
INPUT_CSV = os.path.join(PROCESSED_DATA_DIR, "checkpoint_case_base.csv")
OUTPUT_CSV = os.path.join(PROCESSED_DATA_DIR, "checkpoint_case_base_copy.csv")


def main():

    # Logic for checkpoint 
    if os.path.exists(OUTPUT_CSV):
        print(f"Found existing output file. Resuming progress from: {OUTPUT_CSV}")
        df = pd.read_csv(OUTPUT_CSV)
    else:
        print(f"Starting fresh. Loading initial data from: {INPUT_CSV}")
        try:
            df = pd.read_csv(INPUT_CSV)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the CSV at {INPUT_CSV}")

    # Ensure the target column exists and is strictly typed to accept text strings
    if 'solution_insights' not in df.columns:
        df['solution_insights'] = pd.Series(dtype='object')
    else:
        df['solution_insights'] = df['solution_insights'].astype('object')


    # 2. INITIALIZE MANAGERS
    
    # Initialize the Summarizer in batch mode (Enables the 15 RPM pacing)
    print("Initializing Gemma 31B Synthesizer...")
    synthesizer = SynthesizerManager(provider="gemma_31b", is_batch=True)

    # Initialize the Hallucination Manager
    print("Initializing Hallucination Detection Manager (Jina(M) -> Eigen(S))...")
    jina_method = JinaClipEvaluator() 
    eigen_method = EigenScoreEvaluator()
    hallucination_manager = HallucinationManager(
        cross_modal_method=jina_method, 
        cross_output_method=eigen_method, 
        aggregation_style="sequential_modal_first",
        cross_modal_style="sequential",
        thresholds={
            "modal_primary": 0.35,   # Medium tier for Jina
            "modal_secondary": 0.00, # Unused in this config
            "output": 0.94           # Strict tier for Eigen
        }
    )


    # 3. BATCH PROCESSING LOOP
    
    print("Starting processing loop...")
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Cases"):
        
        # Skip this row if it has already been processed
        if pd.notna(row.get('solution_insights')):
            continue
            
        # Safely extract the texts, ignoring actual NaNs or empty strings
        raw_texts = [
            str(row.get('qwen_insight', '')),
            str(row.get('pixtral_insight', '')),
            str(row.get('idefics_insight', ''))
        ]
        
        # Filter out "nan" strings or empty spaces just in case a model failed to output
        texts = [t for t in raw_texts if t.strip() and t.lower() != 'nan']
        
        # Construct the absolute path for the image
        img_path = os.path.join(RAW_DATA_DIR, str(row.get('img_path', '')))
        
        # STEP 1: Hallucination Detection
        surviving_texts, report = hallucination_manager.filter_outputs(img_path, texts)
        
        # STEP 2: Fallback Logic
        if not surviving_texts:

            # Identify the output with the highest visual grounding score (Jina)
            # We rely on Jina-CLIP as the unique fallback method following the hallucination experiment results
            jina_result = jina_method.evaluate(img_path, texts)
            best_idx = jina_result.scores.index(max(jina_result.scores))
            best_text = texts[best_idx]
            
            # Force the highest-scoring text through
            surviving_texts = [best_text]
        
        # STEP 3: Synthesis
        try:
            final_insight = synthesizer.process(surviving_texts)
        except Exception as e:
            print(f"\nSynthesizer Error at index {index} (ID: {row.get('id')}): {e}")
            # We save the file immediately before crashing so progress isn't lost
            df.to_csv(OUTPUT_CSV, index=False)
            raise e
                
        # Update the dataframe
        df.at[index, 'solution_insights'] = final_insight
        
        # STEP 4: Checkpointing
        if index % 20 == 0:
            df.to_csv(OUTPUT_CSV, index=False)

        # STEP 5: Memory Cleanup
        
        gc.collect() # Force garbage collection of unreferenced Python objects
        
        # Force PyTorch to release cached VRAM back to the laptop
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final save when the loop finishes completely
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nBatch processing complete! Output saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()