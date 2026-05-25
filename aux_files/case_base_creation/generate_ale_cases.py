import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from PyALE import ale
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

CSV_FILE = 'xai_case_base.csv'

IMGS_FOLDER = 'xai_images/'

COLUMNS = [
    'id', 'img_path', 'domain', 'ai_task', 'ai_problem_type', 'class', 
    'library', 'input_format', 'xai_graph_type', 'ai_model', 'explainer', 
    'attributes', 'scope', 'portability', 'concurrency', 'description'
]

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

    explainer_name = type(explainer).__name__ if explainer else "PyALE_Function"
    
    # ALE is an Agnostic explainer by nature

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

def generate_ale_plots(model, X, domain, ai_task, ai_problem_type, vars_1d, vars_2d, target_class_name="NA", target_class_idx=None):
    
    """Recieves any model and data, generates all ALE plots and saves them in the CSV"""

    # Path for saving images (we create the folder if it doesn't exist)

    os.makedirs(IMGS_FOLDER, exist_ok=True)
    
    # Class string logic (for local regression the predicted value, for classification the target class)

    if ai_problem_type.lower() == "regression":

        # As ALE is generally global by nature, for regression we can just use "NA"

        global_class_str = "NA"

    else:

        # For classification, it remains the target class across the board

        global_class_str = "NA" if target_class_idx is None else target_class_name

    # Defaults for CSV

    library = "ALE"

    input_format = "tabular"

    # PyALE strictly demands an object with a .predict() method. 
    # We create a dynamic wrapper that feeds PyALE exactly what it needs for any problem type.
    class PyALEWrapper:

        def __init__(self, base_model, prob_type, target_col):
            self.base_model = base_model
            self.prob_type = prob_type
            self.target_col = target_col
            
        def predict(self, data):
            if self.prob_type == "regression":
                return self.base_model.predict(data)
            else: # Classification: isolate the specific class probability
                try:
                    probs = self.base_model.predict_proba(data)
                    if len(probs.shape) > 1 and probs.shape[1] > 1:
                        return probs[:, self.target_col]
                    return probs
                except AttributeError:
                    return self.base_model.predict(data).flatten()
                

    target_col = target_class_idx if target_class_idx is not None else 1
    ale_model_obj = PyALEWrapper(model, ai_problem_type.lower(), target_col)

    # 1. ALE 1D

    for var in vars_1d:   # We loop through the provided 1D features

        current_id = get_next_id()
        img_path = f"{IMGS_FOLDER}{current_id}.png"
        plt.figure()
        ale_eff = ale(   # The 'ale' function generates the plot directly in the current matplotlib figure
            X=X, 
            model=ale_model_obj, 
            feature=[var],
            grid_size=50, 
            include_CI=True   # Include the confidence intervals (shaded areas)
        )
        fig = plt.gcf()   # Get the current figure
        plt.tight_layout()
        plt.savefig(img_path, bbox_inches='tight', facecolor='white')   # White background for better visibility in case of transparency
        plt.close(fig)
        register_case(current_id, img_path, domain, ai_task, ai_problem_type, global_class_str, library, input_format, "1D", model, None, var, "Global", "Post")

    # 2. ALE 2D (interaction of two variables)

    for vars_pair in vars_2d:   # We loop through the provided 2D feature pairs

        current_id = get_next_id()
        img_path = f"{IMGS_FOLDER}{current_id}.png"
        plt.figure()
        ale_eff_2d = ale(   # The 'ale' function generates the plot directly in the current matplotlib figure
            X=X, 
            model=ale_model_obj, 
            feature=vars_pair,   
            grid_size=20
        )
        fig_2d = plt.gcf()   # Get the current figure
        plt.tight_layout()
        plt.savefig(img_path, bbox_inches='tight', facecolor='white')   # White background for better visibility in case of transparency
        plt.close(fig_2d)
        register_case(current_id, img_path, domain, ai_task, ai_problem_type, global_class_str, library, input_format, "2D", model, None, ", ".join(vars_pair), "Global", "Post")

# MAIN BLOCK (Modularized for easy reuse with different models/datasets)

if __name__ == "__main__":
    
    # A. We load de data (changeble for any other dataset)

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
    data = pd.read_csv(url)
    
    # Map categorical time variables to integers so models can process them
    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    day_map = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}
    
    data['month'] = data['month'].map(month_map)
    data['day'] = data['day'].map(day_map)

    X = data.drop(columns=['area'])
    y = data['area']

    # The chosen feature pairs for Forest Fires
    fire_1d = ['temp', 'wind']
    fire_2d = [['temp', 'wind'], ['temp', 'RH']]

    # B. We train a base model (changable for any other model)

    # =========================================================
    # DEEP LEARNING DATA SCALING
    # =========================================================
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    X_tf, y_tf = X_scaled.values, y.values
    X_pt = torch.tensor(X_scaled.values, dtype=torch.float32)
    y_pt = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

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
        loss = criterion(pytorch_model(X_pt), y_pt)
        loss.backward()
        optimizer.step()

    pytorch_model.eval()

    # Wrapper so PyALE can use PyTorch like a standard regressor
    class TorchRegressor:
        def __init__(self, model):
            self.model = model
        def predict(self, input_data):
            if isinstance(input_data, pd.DataFrame): input_data = input_data.values
            tensor_data = torch.tensor(input_data, dtype=torch.float32)
            with torch.no_grad():
                return self.model(tensor_data).numpy().flatten()

    pytorch_wrapper = TorchRegressor(pytorch_model)

    generate_ale_plots(
        model=pytorch_wrapper, 
        X=X_scaled, 
        domain="Environment", 
        ai_task="Forest fire burned area prediction",
        ai_problem_type="Regression", 
        vars_1d=fire_1d,
        vars_2d=fire_2d,
        target_class_idx=None
    )

    # C. We call the modular function to generate and register all ALE plots

    # generate_ale_plots(
    #     model=model, 
    #     X=X, 
    #     domain="Healthcare",
    #     ai_task="Diabetes progression prediction",
    #     ai_problem_type="Regression",
    #     target_class_idx=None
    # )