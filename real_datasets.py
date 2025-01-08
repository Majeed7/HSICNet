from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import numpy as np 

from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils.multiclass import type_of_target

from sklearn.svm import SVC, SVR
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils.multiclass import type_of_target

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

# Fetch arrhythmia dataset from OpenML (if available)
arrhythmia = fetch_openml(name="arrhythmia", version=1, as_frame=True)
X, y = arrhythmia.data, arrhythmia.target
print(X.shape, y.shape)

# Fetch Musk dataset from OpenML
# musk = fetch_openml(name="musk", version=1, as_frame=True)
# X, y = musk.data, musk.target
# print(X.shape, y.shape)

# X = X.drop(columns=['molecule_name'])

# Fetch Madelon dataset
madelon = fetch_openml(name="madelon", version=1, as_frame=True)
X, y = madelon.data, madelon.target
print(X.shape, y.shape)

# Fetch nomao dataset from OpenML
nomao = fetch_openml(name="nomao", version=1, as_frame=True)
X, y = nomao.data, nomao.target

# Fetch gisette dataset from OpenML
gisette = fetch_openml(name="gisette", version=1, as_frame=True)
X, y = gisette.data, gisette.target


#def sonar():
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
df = pd.read_csv(url, header=None, sep=',')


waveform = fetch_openml(name="waveform-5000", version=1, as_frame=True)
X, y = waveform.data, waveform.target

# Fetch steel-plates-fault dataset from OpenML  
steel = fetch_openml(name="steel-plates-fault", version=1, as_frame=True)
X, y = steel.data, steel.target

steel = fetch_openml(name="steel-plates-fault", version=1, as_frame=True)
X, y = steel.data, steel.target


santander = fetch_openml(name="SantanderCustomerSatisfaction", version=1, as_frame=True)
X_santander, y_santander = santander.data, santander.target

'''
Begining with the training of the model 
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=500,  # Number of trees
    max_depth=10,    # Maximum depth of trees
    random_state=42    # For reproducibility
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
inf_feature = model.feature_importances_
# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
#print("\nClassification Report:\n", classification_report(y_test, y_pred))


plt.figure(figsize=(8, 6))
fnames = [f'Feature {i}' for i in range(X.shape[1])]
plt.barh(fnames, inf_feature, color='skyblue')


# Calculate the threshold for the top 5% most important features
threshold = np.percentile(inf_feature, 95)  # 95th percentile
# Get the indices of the top 5% most important features
top_5_percent_indices = np.where(inf_feature >= threshold)[0]

X_train_sel = X_train.iloc[:, top_5_percent_indices]
X_test_sel = X_test.iloc[:, top_5_percent_indices]
model.fit(X_train_sel, y_train)

y_pred = model.predict(X_test_sel)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
#print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("done!")