import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
import numpy as np
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
from keras.models import Sequential    # tensorflow.keras.models import Sequential
from keras.layers import Dense, Input  # tensorflow.keras.layers import Dense, Input
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


CSV_FILE = 'xai_case_base.csv'

IMGS_FOLDER = 'xai_images/'

COLUMNS = [
    'id', 'img_path', 'domain', 'ai_task', 'ai_problem_type', 'class', 
    'library', 'input_format', 'xai_graph_type', 'ai_model', 'explainer', 
    'attributes', 'scope', 'portability', 'concurrency', 'description'
]

SPECIFIC_EXPLAINERS = ['TreeExplainer', 'DeepExplainer', 'GradientExplainer', 'LinearExplainer']


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
    
    # Depending on the explainer type, we can determine portability (Model-Specific/Agnostic)

    portability = "Specific" if explainer_name in SPECIFIC_EXPLAINERS else "Agnostic"

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


def generate_shap_plots(model, explainer, shap_values, X, domain, ai_task, ai_problem_type, target_class_name="NA", target_class_idx=None):

    """Recieves any model, explainer, pre-calculated SHAP values and data, generates all SHAP plots and saves them in the CSV"""
    
    # Path for saving images (we create the folder if it doesn't exist)

    os.makedirs(IMGS_FOLDER, exist_ok=True)

    # If it's a multiclass problem and a target class is specified, filter

    if target_class_idx is not None and len(shap_values.shape) == 3:
        
        shap_values = shap_values[:, :, target_class_idx]

    # Extract base value dynamically (according to the target class) to generalize across modern and older SHAP models

    # if hasattr(explainer, "expected_value"):   # For TreeExplainer, LinearExplainer, KernelExplainer...

    #     base_value = explainer.expected_value

    #     # If explainer.expected_value is a list/array (multiclass), select the target index

    #     if target_class_idx is not None and isinstance(base_value, (list, tuple, np.ndarray)):

    #         if len(base_value) > target_class_idx: # Normal behavior (e.g., Random Forest returns 2 values)
                
    #             base_value = base_value[target_class_idx]

    #         else: # Edge case (e.g., GradientBoosting binary only returns 1 value)
                
    #             base_value = base_value[0]

    # else: # For PermutationExplainer, ExactExplainer...

    #     # We choose the first prediction's base value (it is usually the same for all samples)
    #     # In new explainer versions, base_value already adapts to the filtered shap_values

    #     base_value = shap_values.base_values[0]

    base_value = shap_values.base_values[0]

    if isinstance(base_value, (list, np.ndarray, pd.Series)):

        base_value = float(np.squeeze(base_value))

    # Class string logic (for local regression the predicted value, for classification the target class)

    if ai_problem_type.lower() == "regression":

        global_class_str = "NA"

        # Calculate the exact prediction for the single instance (X.iloc[0]) used in local plots

        local_pred = float(base_value + shap_values[0].values.sum())

        local_class_str = f"{local_pred:.2f}"    # Formatting to 2 decimals for readability

    else:

        # For classification, it remains the target class across the board

        global_class_str = "NA" if target_class_idx is None else target_class_name

        local_class_str = global_class_str

    # Defaults for CSV

    all_attributes = ", ".join(X.columns.tolist())

    library = "SHAP"

    input_format = "tabular"
    

    ##### PLOT GENERATION #####

    # 1. Waterfall Plot (Local - Instance 0)

    current_id = get_next_id()
    img_path = f"{IMGS_FOLDER}{current_id}.png"
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.savefig(img_path, bbox_inches='tight', facecolor='white')   # White background for better visibility in case of transparency
    plt.close()
    register_case(current_id, img_path, domain, ai_task, ai_problem_type, local_class_str, library, input_format, "Waterfall", model, explainer, all_attributes, "Local", "Post")

    # 2. Force Plot (Local - Instance 0)

    current_id = get_next_id()
    img_path = f"{IMGS_FOLDER}{current_id}.png"
    plt.figure()
    shap.force_plot(base_value, shap_values.values[0], X.iloc[0], matplotlib=True, show=False)   # We use matplotlib=True to get a static image
    plt.savefig(img_path, bbox_inches='tight', facecolor='white')   # White background for better visibility in case of transparency
    plt.close()
    register_case(current_id, img_path, domain, ai_task, ai_problem_type, local_class_str, library, input_format, "Force", model, explainer, all_attributes, "Local", "Post")

    # 3. Bar Plot

    # 3.1 Bar Plot (Local - Instance 0)

    current_id = get_next_id()
    img_path = f"{IMGS_FOLDER}{current_id}.png"
    plt.figure()
    shap.plots.bar(shap_values[0], show=False)
    plt.savefig(img_path, bbox_inches='tight', facecolor='white')   # White background for better visibility in case of transparency
    plt.close()
    register_case(current_id, img_path, domain, ai_task, ai_problem_type, local_class_str, library, input_format, "Bar", model, explainer, all_attributes, "Local", "Post")

    # 3.2 Bar Plot (Cohort)

    current_id = get_next_id()
    img_path = f"{IMGS_FOLDER}{current_id}.png"
    plt.figure()
    # We create a dictionary separating in two groups to compare them (it could be any other separation such as by a feature value, e.g. X['age'] > 50)
    generic_cohorts = {
        "Group A": shap_values[:len(shap_values)//2],
        "Group B": shap_values[len(shap_values)//2:]
    }
    shap.plots.bar(generic_cohorts, show=False)
    plt.savefig(img_path, bbox_inches='tight', facecolor='white')   # White background for better visibility in case of transparency
    plt.close()
    register_case(current_id, img_path, domain, ai_task, ai_problem_type, global_class_str, library, input_format, "Bar", model, explainer, all_attributes, "Cohort", "Post")

    # 3.3 Bar Plot (Global)

    current_id = get_next_id()
    img_path = f"{IMGS_FOLDER}{current_id}.png"
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.savefig(img_path, bbox_inches='tight', facecolor='white')   # White background for better visibility in case of transparency
    plt.close()
    register_case(current_id, img_path, domain, ai_task, ai_problem_type, global_class_str, library, input_format, "Bar", model, explainer, all_attributes, "Global", "Post")

    # 4. Beeswarm Plot (Global)

    current_id = get_next_id()
    img_path = f"{IMGS_FOLDER}{current_id}.png"
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig(img_path, bbox_inches='tight', facecolor='white')   # White background for better visibility in case of transparency
    plt.close()
    register_case(current_id, img_path, domain, ai_task, ai_problem_type, global_class_str, library, input_format, "Beeswarm", model, explainer, all_attributes, "Global", "Post")

    # 5. Violin Plot (Global)

    current_id = get_next_id()
    img_path = f"{IMGS_FOLDER}{current_id}.png"
    plt.figure()
    shap.plots.violin(shap_values, show=False)
    plt.savefig(img_path, bbox_inches='tight', facecolor='white')   # White background for better visibility in case of transparency
    plt.close()
    register_case(current_id, img_path, domain, ai_task, ai_problem_type, global_class_str, library, input_format, "Violin", model, explainer, all_attributes, "Global", "Post")

    # 6. Scatter Plot (Global; also called Dependence - using the most important variable through global feature importance orderings (explanation.abs.mean(0).argsort[-1]]) )

    current_id = get_next_id()
    img_path = f"{IMGS_FOLDER}{current_id}.png"
    most_important_idx = shap_values.abs.mean(0).values.argsort()[-1]   # Index of the most important variable
    plt.figure()
    shap.plots.scatter(shap_values[:, most_important_idx], color=shap_values, show=False)
    plt.savefig(img_path, bbox_inches='tight', facecolor='white')   # White background for better visibility in case of transparency
    plt.close()
    register_case(current_id, img_path, domain, ai_task, ai_problem_type, global_class_str, library, input_format, "Scatter", model, explainer, X.columns[most_important_idx], "Global", "Post")

    # 7. Heatmap Plot (Global)

    current_id = get_next_id()
    img_path = f"{IMGS_FOLDER}{current_id}.png"
    plt.figure()
    shap.plots.heatmap(shap_values, show=False)
    plt.savefig(img_path, bbox_inches='tight', facecolor='white')   # White background for better visibility in case of transparency
    plt.close()
    register_case(current_id, img_path, domain, ai_task, ai_problem_type, global_class_str, library, input_format, "Heatmap", model, explainer, all_attributes, "Global", "Post")

    # 8. Decision Plot (Cohort / Global) [we use 20 instances for better visualization]

    current_id = get_next_id()
    img_path = f"{IMGS_FOLDER}{current_id}.png"
    plt.figure()
    shap.decision_plot(base_value, shap_values.values[:20], X.iloc[:20], show=False)   # We need to extract the expected_value and the .values explicitly for this plot
    plt.savefig(img_path, bbox_inches='tight', facecolor='white')   # White background for better visibility in case of transparency
    plt.close()
    register_case(current_id, img_path, domain, ai_task, ai_problem_type, global_class_str, library, input_format, "Decision", model, explainer, all_attributes, "Cohort", "Post")

# MAIN BLOCK (Modularized for easy reuse with different models/datasets)

if __name__ == "__main__":

    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    # warnings.filterwarnings('ignore') # Ignores convergence warnings
    
    # A. We load de data (changeble for any other dataset)

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
    data = pd.read_csv(url)
    
    # Map categorical time variables to integers so Deep Learning can process them
    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    day_map = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}
    
    data['month'] = data['month'].map(month_map)
    data['day'] = data['day'].map(day_map)

    X = data.drop(columns=['area'])
    y = data['area']

    # --- CRUCIAL: Scale the data for Deep Learning ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Data for Keras (Numpy Arrays - Scaled)
    X_tf, y_tf = X_scaled, y.values

    # Data for PyTorch (Tensors - Scaled)
    X_pt = torch.tensor(X_scaled, dtype=torch.float32)
    y_pt = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    # B. We train a base model (changable for any other model)

    # =========================================================
    # COMBINATION 2: PYTORCH + GRADIENT EXPLAINER
    # =========================================================
    class TorchRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(X.shape[1], 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 1) # No activation
            )
        def forward(self, x): 
            return self.net(x)

    pytorch_model = TorchRegressor()
    optimizer = optim.Adam(pytorch_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for _ in range(100):
        optimizer.zero_grad()
        loss = criterion(pytorch_model(X_pt), y_pt)
        loss.backward()
        optimizer.step()

    pytorch_model.eval() 
    
    explainer_grad = shap.GradientExplainer(pytorch_model, X_pt)
    
    # Process SHAP values robustly
    raw_grad_output = explainer_grad.shap_values(X_pt)
    if isinstance(raw_grad_output, (list, tuple)): raw_grad_output = raw_grad_output[0]
    if hasattr(raw_grad_output, 'shape') and len(raw_grad_output.shape) > 2: 
        raw_grad_output = np.squeeze(raw_grad_output)
    
    # Process Base Value
    base_grad_val = pytorch_model(X_pt).mean().item()

    # Package into SHAP object for plotting
    shap_values_pytorch = shap.Explanation(
        values=raw_grad_output, 
        base_values=np.full(X.shape[0], base_grad_val), 
        data=X_scaled, 
        feature_names=X.columns.tolist()
    )

    generate_shap_plots(
        model=pytorch_model, 
        explainer=explainer_grad, 
        shap_values=shap_values_pytorch,
        X=X, 
        domain="Environment",
        ai_task="Forest fire burned area prediction",
        ai_problem_type="Regression",
        target_class_idx=None
    )

    # C. Choose explainer and calculate SHAP values explicitly



    # D. We call the modular function to generate and register all SHAP plots