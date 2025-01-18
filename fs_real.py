from sklearn.datasets import fetch_openml
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

import os
import pickle
import time
from openpyxl import Workbook
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest, f_classif, f_regression, RFECV
from sklearn.linear_model import Lasso
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.svm import SVC, SVR
from pathlib import Path
import gc
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from pyHSICLasso import HSICLasso

from real_datasets import load_dataset

from HSICNet.HSICNet import *
from HSICNet.HSICFeatureNet import *

import sys 

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

# Ensure a directory exists for saving models
os.makedirs("trained_models", exist_ok=True)

# Define the list of feature selectors
feature_selectors = ["HSICFeatureNetGumbelSparsemax", "HSICNetGumbelSparsemax", "HSICLasso", "mutual_info", "lasso", "k_best", "tree_ensemble"]

# Initialize an Excel workbook to store global importance values
wb = Workbook()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def memory_cleaning():
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()

epoch=500
layers = [200, 300, 400, 300, 200]
feature_layers = [20, 50, 100, 50, 20]
act_fun_featlayer = torch.nn.SELU
act_fun_layer = torch.nn.Sigmoid

if __name__ == '__main__':
        
    dataset_names1 = ["breast_cancer", "sonar", "nomao", "steel"] #  
    dataset_names2 = ["breast_cancer_wisconsin", "skillcraft", "ionosphere", "sml", "pol"]
    dataset_names3 = ['parkinson', 'keggdirected', "pumadyn32nm", "crime", "gas", "waveform"]
    dataset_names4 = ['autos', 'bike', 'keggundirected']

        # Check if an argument is passed
    if len(sys.argv) < 2:
        print("No argument passed. Using default value: 1")
        ds_index = 1
        dataset_names = dataset_names1
    else:
        # Retrieve the parameter passed from Bash
        parameter = sys.argv[1]

        # Try converting the argument to a number
        try:
            # Try converting to an integer
            ds_index = int(parameter)

            if ds_index == 1:
                dataset_names = dataset_names1
            elif ds_index == 2:
                dataset_names = dataset_names2
            elif ds_index == 3:
                dataset_names = dataset_names3
            elif ds_index == 4:
                dataset_names = dataset_names4

        except ValueError:
            # If it fails, try converting to a float
            ds_index = 1
            dataset_names = dataset_names1
            print("Cannot process the value. Using default value: 0.1")


    # Main running part of the script
    for dataset_name in dataset_names:
        print(f"\nProcessing dataset: {dataset_name}")
        try:
            X, y = load_dataset(dataset_name)
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            continue

        # Determine if the dataset is for classification or regression
        mode = "classification" if type_of_target(y) in ["binary", "multiclass"] else "regression"
        if mode == "classification": continue 

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        n, d = X_train.shape

        # Convert the data to PyTorch tensors
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device=device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).to(device=device)
        sigma_init_X = torch.tensor([0.5]*d, device=device) #initialize_sigma_median_heuristic(X_tensor)
        sigma_init_Y = torch.tensor(0.5, device=device) #initialize_sigma_y_median_heuristic(y_tensor)

        # Get the most representative samples for the dataset
        n_clusters = np.min((500, n))# Number of clusters (desired subset size)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_train)
        centroids = kmeans.cluster_centers_

        closest_indices, _ = pairwise_distances_argmin_min(centroids, X_train)

        X_gx = X_tensor[closest_indices]
        y_gx = y_tensor[closest_indices]

        # Prepare an Excel sheet for the current dataset
        sheet = wb.create_sheet(title=dataset_name)
        sheet.append(["Feature Selector", "Execution Time"] + [f"Feature {i}" for i in range(X.shape[1])])


        # Apply each feature selector
        for selector in feature_selectors:
            print(f"Applying feature selector: {selector} on dataset: {dataset_name}")
            start_time = time.time()

            if selector == "HSICFeatureNetGumbelSparsemax":
                featuregumbelsparsemax_model = HSICFeatureNetGumbelSparsemax(
                    d, feature_layers, act_fun_featlayer, layers,
                    act_fun_layer, sigma_init_X, sigma_init_Y, num_samples=10, temperature=20).to(device=device)
                featuregumbelsparsemax_model.train_model(X_tensor, y_tensor, num_epochs=epoch, BATCH_SIZE=200)
                weights = featuregumbelsparsemax_model(X_gx)[0]
                hsicfngs_sv, v0 = featuregumbelsparsemax_model.global_shapley_value(
                    X_gx, y_gx, featuregumbelsparsemax_model.sigmas, featuregumbelsparsemax_model.sigma_y, weights
                )
                global_importance = hsicfngs_sv.detach().cpu().numpy().squeeze()
                model_filename = f"trained_models/hsicfeaturegumbelsparsemax_{dataset_name}.pkl"
                with open(model_filename, "wb") as f:
                    pickle.dump(featuregumbelsparsemax_model, f)
                del featuregumbelsparsemax_model
                memory_cleaning()

            elif selector == "HSICNetGumbelSparsemax":
                gumbelsparsemax_model = HSICNetGumbelSparsemax(
                    d, layers, act_fun_layer, sigma_init_X, sigma_init_Y, num_samples=10, temperature=20).to(device=device)
                gumbelsparsemax_model.train_model(X_tensor, y_tensor, num_epochs=epoch, BATCH_SIZE=200)
                weights = gumbelsparsemax_model(X_gx)[0]
                hsicgs_sv, v0 = gumbelsparsemax_model.global_shapley_value(
                    X_gx, y_gx, gumbelsparsemax_model.sigmas, gumbelsparsemax_model.sigma_y, weights)
                global_importance = hsicgs_sv.detach().cpu().numpy().squeeze()
                model_filename = f"trained_models/hsicnetgumbelsparsemax_{dataset_name}.pkl"
                with open(model_filename, "wb") as f:
                    pickle.dump(gumbelsparsemax_model, f)
                del gumbelsparsemax_model
                memory_cleaning()

            elif selector == "HSICLasso":
                hsic_lasso = HSICLasso()
                hsic_lasso.input(X_train,y_train.squeeze())
                if mode == "classification": hsic_lasso.classification(d, covars=X_train) 
                else: hsic_lasso.regression(d, covars=X_train)
                hsic_ind = hsic_lasso.get_index()
                init_ranks = (len(hsic_ind) + (d - 1/2 - len(hsic_ind))/2) * np.ones((d,))
                init_ranks[hsic_ind] = np.arange(1,len(hsic_ind)+1)
                global_importance = d - init_ranks 

            elif selector == "mutual_info":
                global_importance = mutual_info_classif(X_train, y_train) if mode == "classification" else mutual_info_regression(X_train, y_train)

            elif selector == "lasso":
                lasso = Lasso().fit(X_train, y_train)
                global_importance = np.abs(lasso.coef_)

            elif selector == "rfecv":
                estimator = SVC(kernel="linear") if mode == "classification" else SVR(kernel="linear")
                rfecv = RFECV(estimator, step=1, cv=5)
                rfecv.fit(X_train, y_train)
                global_importance = rfecv.ranking_

            elif selector == "k_best":
                bestfeatures = SelectKBest(score_func=f_classif, k="all") if mode == "classification" else SelectKBest(score_func=f_regression, k="all")
                fit = bestfeatures.fit(X_train, y_train)
                global_importance = fit.scores_

            elif selector == "tree_ensemble":
                model = ExtraTreesClassifier(n_estimators=50) if mode == "classification" else ExtraTreesRegressor(n_estimators=50)
                model.fit(X_train, y_train)
                global_importance = model.feature_importances_

            else:
                print(f"Unknown feature selector: {selector}")
                continue

            execution_time = time.time() - start_time
            print(f"Execution time for {selector}: {execution_time}")

            # Store global importance values in the Excel sheet
            sheet.append([selector, execution_time] + list(global_importance))

        # Save the Excel file after processing each dataset
        excel_filename = f"real_fs_feature_importance_{ds_index}.xlsx"
        wb.save(excel_filename)
        print(f"Global feature importance for {dataset_name} saved to {excel_filename}")
    
    wb.close()
    print("All datasets processed!")
    