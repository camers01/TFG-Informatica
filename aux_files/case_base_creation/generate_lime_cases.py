import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import lime
import lime.lime_tabular
from sklearn.datasets import load_diabetes, fetch_california_housing, load_wine, fetch_openml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor # Red neuronal simple
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential as KerasSequential    # tensorflow.keras.models import Sequential
from keras.layers import Dense, Input  # tensorflow.keras.layers import Dense, Input
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from html2image import Html2Image

CSV_FILE = 'xai_case_base.csv'

IMGS_FOLDER = 'xai_images/'

COLUMNS = [
    'id', 'img_path', 'domain', 'ai_task', 'ai_problem_type', 'class', 
    'library', 'input_format', 'xai_graph_type', 'ai_model', 'explainer', 
    'attributes', 'scope', 'portability', 'concurrency', 'description'
]

hti = Html2Image(output_path=IMGS_FOLDER)   # Initialize HTML screenshot tool

def get_next_id():

    """Reads the CSV and returns the next available ID in 000000 format"""

    if not os.path.exists(CSV_FILE):
        return "000001"
    
    else:
        df = pd.read_csv(CSV_FILE, keep_default_na=False)
        return f"{len(df) + 1:06d}"


def register_case(id_case, img_path, domain, ai_task, ai_problem_type, analyzed_class, library, input_format, xai_graph_type, model, explainer, attributes, scope, concurrency, description=""):
    
    """Saves the record in the CSV"""
    
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=COLUMNS)

    else:
        df = pd.read_csv(CSV_FILE, keep_default_na=False)

    # Dynamic extraction of model and explainer names

    model_name = type(model).__name__

    explainer_name = type(explainer).__name__
    
    # LIME is an Agnostic explainer by nature

    portability = "Agnostic"

    new_row = {
        'id': id_case,
        'img_path': img_path,
        'domain': domain,
        'ai_task': ai_task,
        'ai_problem_type': ai_problem_type,
        'class': analyzed_class,
        'library': library,
        'input_format': input_format,
        'xai_graph_type': xai_graph_type,
        'ai_model': model_name,
        'explainer': explainer_name,
        'attributes': attributes,
        'scope': scope,
        'portability': portability,
        'concurrency': concurrency,
        'description': description 
    }
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df['id'] = df['id'].astype(int).apply(lambda x: f"{x:06d}")   # Force pandas to not convert to int and keep the leading zeros 

    df.to_csv(CSV_FILE, index=False)


def generate_lime_plots(model, X_train, X_test, domain, ai_task, ai_problem_type, target_class_name="NA", target_class_idx=None):
    
    """Recieves any model and data, generates all LIME plots and saves them in the CSV"""
    
    # Path for saving images (we create the folder if it doesn't exist)
    
    os.makedirs(IMGS_FOLDER, exist_ok=True)
    
    # Determine LIME mode based on problem type

    lime_mode = "regression" if ai_problem_type.lower() == "regression" else "classification"
    
    # Initialize Explainer

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        mode=lime_mode,
        random_state=42
    )
    
    # Choose the instance to explain (first row of testing set)

    instance_idx = 0

    instance_to_explain = X_test.iloc[instance_idx]
    
    # Class string logic (for local regression the predicted value, for classification the target class)
    
    if ai_problem_type.lower() == "regression":

        # Calculate exact prediction for for the single instance (X.iloc[instance_idx]) in local regression

        local_pred = model.predict(X_test.iloc[[instance_idx]])[0]

        local_class_str = f"{local_pred:.2f}"   # Formatting to 2 decimals for readability

    else:

        # For classification, it remains the target class across the board
            
        local_class_str = "NA" if target_class_idx is None else target_class_name

    # Set up prediction function for LIME

    if lime_mode == "regression":
        def predict_fn(data_array):
            return model.predict(pd.DataFrame(data_array, columns=X_train.columns))
        label_to_plot = 1 # LIME ignores this for regression
    else:
        def predict_fn(data_array):
            return model.predict_proba(pd.DataFrame(data_array, columns=X_train.columns))
        # Ensure we tell LIME exactly which class to explain
        label_to_plot = target_class_idx if target_class_idx is not None else 1

    # Generate explanation

    exp = explainer.explain_instance(
        data_row=instance_to_explain.values, 
        predict_fn=predict_fn,
        labels=(label_to_plot,) # Force LIME to explain the target class
    )

    # Defaults for CSV

    all_attributes = ", ".join(X_train.columns.tolist())

    library = "LIME"

    input_format = "tabular"
    
    ##### PLOT GENERATION #####

    # 1. Matplotlib Bar Chart 
    
    current_id = get_next_id()
    img_path = f"{IMGS_FOLDER}{current_id}.png"
    fig = exp.as_pyplot_figure(label=label_to_plot)
    plt.tight_layout()
    plt.savefig(img_path, bbox_inches='tight', facecolor='white')   # White background for better visibility in case of transparency
    plt.close(fig)
    register_case(current_id, img_path, domain, ai_task, ai_problem_type, local_class_str, library, input_format, "Bar", model, explainer, all_attributes, "Local", "Post")
    
    # 2. Dashboard HTML (pasamos a PNG a través de captura con html2image)

    current_id = get_next_id()
    img_name = f"{current_id}.png"   # hti uses the name, the path is handled by output_path
    img_path = f"{IMGS_FOLDER}{img_name}"

    # First we save the dashboard as a temporary HTML file and modify it to have a white background to avoid transparency issues
    html_temp = f"{IMGS_FOLDER}temp_{current_id}.html"
    exp.save_to_file(html_temp, labels=[label_to_plot], show_table=True, show_all=False)
    with open(html_temp, 'r', encoding='utf-8') as f:
        html_data = f.read()
    with open(html_temp, 'w', encoding='utf-8') as f:
        f.write(html_data + "<style>body { background-color: white !important; }</style>")
    
    # Second we take a screenshot of the HTML file to convert it to PNG
    hti.screenshot(html_file=html_temp, save_as=img_name, size=(950, 600))
    
    # Finally we clean up the temporary HTML file and register the case in the CSV
    if os.path.exists(html_temp):
        os.remove(html_temp)
    register_case(current_id, img_path, domain, ai_task, ai_problem_type, local_class_str, library, input_format, "Dashboard", model, explainer, all_attributes, "Local", "Post")

# MAIN BLOCK (Modularized for easy reuse with different models/datasets)

if __name__ == "__main__":

    # A. We load de data (changeble for any other dataset)
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
    data = pd.read_csv(url)
    
    # Map categorical time variables to integers so the models can process them
    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    day_map = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}
    
    data['month'] = data['month'].map(month_map)
    data['day'] = data['day'].map(day_map)

    X = data.drop(columns=['area'])
    y = data['area']

    # LIME specifically requires the training set to build its baseline
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # B. We train a base model (changable for any other model)

    # =========================================================
    # DEEP LEARNING DATA SCALING
    # =========================================================
    # Scale features (temperature, humidity, wind) for optimal gradient descent
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    X_train_tf, y_train_tf = X_train_scaled.values, y_train.values
    X_train_pt = torch.tensor(X_train_scaled.values, dtype=torch.float32)
    y_train_pt = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    # =========================================================
    # PYTORCH + WRAPPER
    # =========================================================
    class RealTorchRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(X.shape[1], 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 1) # No activation
            )
        def forward(self, x): 
            return self.net(x)

    pytorch_model = RealTorchRegressor()
    optimizer = optim.Adam(pytorch_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for _ in range(100):
        optimizer.zero_grad()
        loss = criterion(pytorch_model(X_train_pt), y_train_pt)
        loss.backward()
        optimizer.step()

    pytorch_model.eval()

    # Wrapper to make PyTorch behave exactly like a Scikit-Learn regressor for LIME
    class TorchRegressor:
        def __init__(self, model):
            self.model = model
        def predict(self, input_data):
            if isinstance(input_data, pd.DataFrame): input_data = input_data.values
            tensor_data = torch.tensor(input_data, dtype=torch.float32)
            with torch.no_grad():
                return self.model(tensor_data).numpy().flatten()

    pytorch_wrapper = TorchRegressor(pytorch_model)

    generate_lime_plots(
        model=pytorch_wrapper, 
        X_train=X_train_scaled, 
        X_test=X_test_scaled,
        domain="Environment", 
        ai_task="Forest fire burned area prediction",
        ai_problem_type="Regression", 
        target_class_idx=None
    )

    # C. We call the modular function to generate and register all LIME plots

    # generate_lime_plots(
    #     model=model, 
    #     X_train=X_train,
    #     X_test=X_test,
    #     domain="Healthcare",
    #     ai_task="Diabetes progression prediction",
    #     ai_problem_type="Regression",
    #     target_class_idx=None
    # )