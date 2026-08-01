import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import mlflow
import warnings
warnings.filterwarnings("ignore")

# Load the split datasets
X_train = pd.read_csv("Xtrain.csv")
X_test = pd.read_csv("Xtest.csv")
y_train = pd.read_csv("ytrain.csv")['ProdTaken'] # Ensure it's a Series
y_test = pd.read_csv("ytest.csv")['ProdTaken'] # Ensure it's a Series

# Define preprocessing steps
numerical_features = ['Age', 'NumberOfPersonVisiting', 'NumberOfTrips', 
                      'NumberOfChildrenVisiting', 'MonthlyIncome', 'PitchSatisfactionScore', 
                      'NumberOfFollowups', 'DurationOfPitch']
categorical_features = ['TypeofContact', 'CityTier', 'Occupation', 'Gender', 
                        'PreferredPropertyStar', 'MaritalStatus', 'Designation', 'ProductPitched', 'Passport', 'OwnCar']

# Create a column transformer for preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(), numerical_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define the model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

# Define hyperparameter grid for GridSearchCV
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7]
}

# Set up MLflow tracking
mlflow.set_experiment("Tourism Package Prediction")

with mlflow.start_run():
    # Log parameters
    mlflow.log_params({key.replace('classifier__', ''): value for key, value in param_grid.items()})

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        model_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2
    )
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Log best parameters
    mlflow.log_params({key.replace('classifier__', ''): value for key, value in grid_search.best_params_.items()})

    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Log metrics
    mlflow.log_metrics({
        "roc_auc": roc_auc,
        "precision": report['1']['precision'],
        "recall": report['1']['recall'],
        "f1_score": report['1']['f1-score']
    })

    print("Best parameters:", grid_search.best_params_)
    print("ROC AUC on test set:", roc_auc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the best model
    model_path = "tourism_project/deployment/best_xgboost_model.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path)

    print(f"Best model saved to {model_path} and logged to MLflow.")
