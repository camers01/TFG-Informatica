import pandas as pd
import os

CSV_FILE = 'xai_case_base.csv'

IMGS_FOLDER = 'xai_images/'

COLUMNS = [
    'id', 'img_path', 'domain', 'ai_task', 'ai_problem_type', 'class', 
    'library', 'input_format', 'xai_graph_type', 'ai_model', 'explainer', 
    'attributes', 'scope', 'portability', 'concurrency', 'description'
]

def initialize_csv():

    """Creates the CSV file with headers if it doesn't exist"""

    if not os.path.exists(CSV_FILE):

        df = pd.DataFrame(columns=COLUMNS)

        df.to_csv(CSV_FILE, index=False)

def new_entry():

    """Prompts for data via console and appends it to the CSV"""
    
    # Read the current CSV to determine the next ID

    df = pd.read_csv(CSV_FILE)

    new_id = f"{len(df) + 1:06d}"
    
    # Prompt for data via console

    img_name = input("Image filename: ")                   # e.g., shap_plot.png

    img_path = f"{IMGS_FOLDER}{img_name}"
    
    domain = input("Domain: ")                             # e.g., Healthcare, Finance

    ai_task = input("AI Task: ")                           # e.g., cancer detection, price prediction

    ai_problem_type = input("AI Problem Type: ")           # e.g., regression, classification

    analyzed_class = input("Class: ")                      # e.g., malignant, local value, or NA

    library = input("Library: ")                           # e.g., SHAP, LIME

    input_format = input("Input Format: ")                 # e.g., tabular, image, time-series, audio, text

    xai_graph_type = input("XAI Graph Type: ")             # e.g., Summary, Bar
        
    ai_model = input("AI Model: ")                         # e.g., RandomForest, NeuralNetwork

    explainer = input("Explainer: ")                       # e.g., TreeExplainer, KernelExplainer

    attributes = input("Attributes (comma-separated): ")   # e.g., MedInc, HousAge...
    
    scope = input("Scope: ")                               # e.g., Local, Global, Cohort

    portability = input("Portability (Model-...): ")       # e.g., Specific, Agnostic

    concurrency = input("Concurrency (...-hoc): ")         # e.g., Ante, Post
    
    # Create the new entry

    new_row = pd.DataFrame([{
        'id': new_id,
        'img_path': img_path,
        'domain': domain,
        'ai_task': ai_task,
        'ai_problem_type': ai_problem_type,
        'class': analyzed_class,
        'library': library,
        'input_format': input_format,
        'xai_graph_type': xai_graph_type,
        'ai_model': ai_model,
        'explainer': explainer,
        'attributes': attributes,
        'scope': scope,
        'portability': portability,
        'concurrency': concurrency,
        'description': ""   # Left empty for to fill with AI output
    }])
    
    # Concatenate and save

    df = pd.concat([df, new_row], ignore_index=True)
    
    # Force the 6-digit string format just in case Pandas tries to convert it to int

    df['id'] = df['id'].astype(int).apply(lambda x: f"{x:06d}")
    
    df.to_csv(CSV_FILE, index=False)

    print(f"[+] Entry {new_id} added successfully!")

if __name__ == "__main__":

    initialize_csv()
    
    while True:

        continue_prompt = input("\nAdd new entry? (y/n): ").lower()

        if continue_prompt != 'y':

            break
        
        new_entry()