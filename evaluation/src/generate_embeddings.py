import os
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai
from google.genai import types

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Get the directory of the current script
EVAL_ROOT = os.path.dirname(SCRIPT_DIR) # Evaluation root directory
PROJECT_ROOT = os.path.dirname(EVAL_ROOT) # Project root directory
XAI_RAG_DIR = os.path.join(PROJECT_ROOT, "xai_rag_cbr") # RAG system root directory
CASE_BASE_CSV = os.path.join(XAI_RAG_DIR, "data", "processed", "case_base.csv") # Case base CSV path
IMAGES_DIR = os.path.join(XAI_RAG_DIR, "data", "raw", "xai_images") # Images folder path
RAG_OUTPUT_CSV = os.path.join(EVAL_ROOT, "data", "raw", "pixtral_rag_insights.csv") # Solution insights CSV path
EMBEDDINGS_DIR = os.path.join(EVAL_ROOT, "data", "embeddings") # Embeddings output directory

# Corresponding names: new folder for embeddings mapped to the corresponding column name in the CSVs to access
TARGETS = {
    "images": None, 
    "qwen": "qwen_insight",      
    "pixtral": "pixtral_insight",
    "idefics": "idefics_insight",
    "summary": "solution_insights", 
    "rag_solution": "insight"
}

# Create the sub-directories for the embeddings inside of the evaluation/data/embeddings folder if they don't exist
for mod in TARGETS.keys():
    os.makedirs(os.path.join(EMBEDDINGS_DIR, mod), exist_ok=True)

# Initialize GenAI Client for Gemini Embedding 2
load_dotenv()
API_KEY_GEMINI = os.getenv("API_KEY_GEMINI")
if not API_KEY_GEMINI:
    raise ValueError("API_KEY_GEMINI not found in .env file.")
client = genai.Client(api_key=API_KEY_GEMINI)
MODEL_ID = "gemini-embedding-2"


# Embedding call wrapper with retry logic
def get_embedding_with_retry(contents, retries=3, backoff_factor=2):
    """Wraps the API call with exponential backoff to handle rate limits."""
    for attempt in range(retries):
        try:
            response = client.models.embed_content(model=MODEL_ID, contents=contents)
            # gemini-embedding-2 outputs a list of floats, so we convert it to a numpy array
            return np.array(response.embeddings[0].values, dtype=np.float32)
        except Exception as e:
            if attempt == retries - 1:
                raise e
            print(f"API Error: {e}. Retrying in {backoff_factor ** attempt}s...")
            time.sleep(backoff_factor ** attempt)


# Main generation function
def run_embedding_generation():

    # Load both dataframes (case base and rag outputs) ensuring strict string types for ID ('000000')
    print("Loading datasets...")
    df_base = pd.read_csv(CASE_BASE_CSV, dtype={'id': str})
    df_base['id'] = df_base['id'].str.zfill(6)
    df_rag = pd.read_csv(RAG_OUTPUT_CSV, dtype={'id': str})
    df_rag['id'] = df_rag['id'].str.zfill(6)
    
    # Merge RAG insights into the main dataframe using the 'id' column
    df_master = pd.merge(df_base, df_rag, on="id", how="inner")
    print(f"Total merged cases to process: {len(df_master)}")
    
    # Main loop to process each case and generate the embeddings for the corresponding image and texts 
    for _, row in tqdm(df_master.iterrows(), total=len(df_master)):

        case_id = str(row['id']).zfill(6)
        
        # 1. Process Image Embedding (we use raw bytes as the images are already saved as static .png files, skipping encoding)
        img_save_path = os.path.join(EMBEDDINGS_DIR, "images", f"{case_id}.npy")
        if not os.path.exists(img_save_path):
            img_target_path = os.path.join(IMAGES_DIR, f"{case_id}.png")
            with open(img_target_path, "rb") as f:
                image_bytes = f.read()  
            img_content = [types.Part.from_bytes(data=image_bytes, mime_type="image/png")]
            img_vec = get_embedding_with_retry(img_content)
            np.save(img_save_path, img_vec)
            
        # 2. Process Text Embeddings (4 Baselines, 1 Solution)
        for category, column_name in TARGETS.items():

            if category == "images": continue # Skip image category as it is already processed
            
            txt_save_path = os.path.join(EMBEDDINGS_DIR, category, f"{case_id}.npy")
            if not os.path.exists(txt_save_path):
                text_content = str(row[column_name])
                # We use the Task Prefix for Asymmetric Cross-Modal Matching (from the options in documentation)
                prompt = f"title: none | text: {text_content}"
                txt_vec = get_embedding_with_retry(prompt)
                np.save(txt_save_path, txt_vec)
                # Small sleep to respect API quotas
                time.sleep(1)

if __name__ == "__main__":
    run_embedding_generation()
    print("Embedding Generation Complete!")