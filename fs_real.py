from sklearn.datasets import fetch_openml
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils.multiclass import type_of_target
import copy 

def train_svm(X_train, y_train, X_test, y_test):
    """
    Train an SVM or SVR model with imputation for missing values.

    Parameters:
        X_train: Training feature matrix
        y_train: Training labels
        X_test: Testing feature matrix
        y_test: Testing labels

    Returns:
        best_model: Trained model after hyperparameter tuning
        best_params: Best hyperparameters from GridSearchCV
        score: Performance score (accuracy for classification, RMSE for regression)
    """

    # Check the type of the target variable
    target_type = type_of_target(y_train)
    is_classification = target_type in ["binary", "multiclass"]

    # Define the parameter grid
    param_grid = {
        "svm__C": [0.1, 1, 10],  # Regularization parameter
    }

    # Choose the model
    model = SVC() if is_classification else SVR()

    # Create a pipeline with an imputer and the SVM/SVR model
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),  # Impute missing values
        ("svm", model)
    ])

    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy" if is_classification else "neg_mean_squared_error")
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    if is_classification:
        score = accuracy_score(y_test, y_pred)
    else:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        score = rmse

    print("Best Parameters:", best_params)
    print("Performance Score:", score)

    return best_model, best_params, score

def load_dataset(name):
    """
    Load dataset by name.

    Parameters:
        name: Name of the dataset

    Returns:
        X, y: Feature matrix and labels
    """
    if name == "arrhythmia":
        dataset = fetch_openml(name="arrhythmia", version=1, as_frame=True)
    elif name == "madelon":
        dataset = fetch_openml(name="madelon", version=1, as_frame=True)
    elif name == "nomao":
        dataset = fetch_openml(name="nomao", version=1, as_frame=True)
    elif name == "gisette":
        dataset = fetch_openml(name="gisette", version=1, as_frame=True)
    elif name == "waveform":
        dataset = fetch_openml(name="waveform-5000", version=1, as_frame=True)
    elif name == "steel":
        dataset = fetch_openml(name="steel-plates-fault", version=1, as_frame=True)
    elif name == "sonar":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
        dataset = pd.read_csv(url, header=None)
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        return X, y
    else:
        raise ValueError(f"Unknown dataset: {name}")

    X, y = dataset.data, dataset.target
    return X, y

# List of dataset names to process
dataset_names = ["arrhythmia", "madelon", "nomao", "gisette", "waveform", "steel", "sonar"]
dataset_names = ["nomao"]
for dataset_name in dataset_names:
    print(f"\nProcessing dataset: {dataset_name}")
    try:
        X, y = load_dataset(dataset_name)
    except Exception as e:
        print(f"Failed to load dataset {dataset_name}: {e}")
        continue

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train SVM on the full dataset
    print("Training SVM on the full dataset...")
    best_model, best_params, full_score = train_svm(X_train, y_train, X_test, y_test)

    # Feature selection using mutual information
    print("Performing feature selection using mutual information...")
    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Compute mutual information scores
    mi_scores = mutual_info_classif(X_train_imputed, y_train, random_state=42)

    top_10_percent_indices = np.argsort(mi_scores)[-int(len(mi_scores) * 0.1):]

    # Subset the top 10% features
    X_train_top = X_train.iloc[:, top_10_percent_indices]
    X_test_top = X_test.iloc[:, top_10_percent_indices]

    # Retrain SVM with the top 10% features using the best hyperparameters
    print(f"Training SVM with top 10% features on dataset: {dataset_name}")
    model = copy.deepcopy(best_model)    
    model.fit(X_train_top, y_train)
    y_pred = model.predict(X_test_top)
    top_score = accuracy_score(y_test, y_pred)

    print(f"Dataset: {dataset_name} - Full Accuracy: {full_score}, Top 10% Features Accuracy: {top_score}")

print("All datasets processed!")
