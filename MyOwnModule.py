#%% Import Needed Libraries
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import uniform, randint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, RandomizedSearchCV

# classifiers and hyperparameter tuning
from sklearn.tree import DecisionTreeClassifier   # For building decision tree models
from sklearn.neighbors import KNeighborsClassifier # For building KNN
from sklearn.linear_model import LogisticRegression # For building LR
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score, mean_absolute_error,
                             mean_squared_error, root_mean_squared_error, r2_score)

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

import importlib

# Try importing root_mean_squared_error (available in sklearn >= 1.6)
if importlib.util.find_spec("sklearn.metrics"):
    try:
        from sklearn.metrics import root_mean_squared_error
    except ImportError:
        # Fallback: define manually if sklearn < 1.6
        def root_mean_squared_error(y_true, y_pred):
            return np.sqrt(mean_squared_error(y_true, y_pred))

#%% 1. Def Check inConsistency 
def check_inconsistency(data, Categorical_Variables, Ordinal_Variables, dependant_name = None):
    if dependant_name != None and data[dependant_name].dtype in ['object', 'category', 'bool']:
        All = list(Categorical_Variables) + list(Ordinal_Variables) + [dependant_name]
    else:
        All = list(Categorical_Variables) + list(Ordinal_Variables) 
    
    for feature in All:
        Item_categories = data[feature].unique()
        print(feature, 'has', len(Item_categories), 'categories. They are: ', Item_categories)
        print('-----------------------------------------------------------------------------------------')
    
    return
    
#%% 2. Define a custom transformer for outlier treatment
class OutlierTrimmer(BaseEstimator, TransformerMixin):
    """
    Adaptive outlier trimmer that selects the appropriate method based on data distribution:
    1. Tests normality using Shapiro-Wilk test
    2. If normal: uses Z-score method (±3 standard deviations)
    3. If skewed: uses IQR method (Q1-1.5*IQR, Q3+1.5*IQR)
    4. Otherwise: uses percentile method (1st and 99th percentiles)
    """
    def __init__(self, normality_threshold = 0.05, skewness_threshold = 0.5):
        self.normality_threshold = normality_threshold
        self.skewness_threshold = skewness_threshold
        self.bounds_ = {}
        self.methods_ = {}
    
    def fit(self, X, y = None):
        X_df = pd.DataFrame(X)
        
        for col in X_df.columns:
            data = X_df[col].dropna()
            
            # Test normality using Shapiro-Wilk test
            _, p_value = stats.shapiro(data)
            if p_value > self.normality_threshold:
                is_normal = True                
            else:
                is_normal = False
            
            # Calculate skewness
            skewness = abs(stats.skew(data))
            is_skewed = skewness > self.skewness_threshold
            
            # Determine method and calculate bounds
            if is_normal:
                # Z-score method
                mean = data.mean()
                std = data.std()
                lower = mean - 3 * std
                upper = mean + 3 * std
                self.methods_[col] = 'Z-score'
                
            elif is_skewed:
                # IQR method
                Q1 = np.nanquantile(data, 0.25)
                Q3 = np.nanquantile(data, 0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                self.methods_[col] = 'IQR'
                
            else:
                # Percentile method
                lower = np.nanquantile(data, 0.01)
                upper = np.nanquantile(data, 0.99)
                self.methods_[col] = 'Percentile'
            
            self.bounds_[col] = (lower, upper)
        
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            if col in self.bounds_:
                lower, upper = self.bounds_[col]
                X_df[col] = np.clip(X_df[col], lower, upper)
        
        return X_df.values
    
    def get_method_info(self):
        """
        Returns a dictionary showing which method was used for each column
        """
        return {col: {'method': self.methods_[col], 'bounds': np.round(self.bounds_[col],3)} for col in self.methods_.keys()}   
    
#%% 3. Preprocessing
def preprocessing(Numerical_variables, Categorical_variables):
    # Define preprocessing for numerical and categorical data
    Numerical_transformer = Pipeline(steps = [('outlier_trimmer', OutlierTrimmer()),     # Outlier treatment
                                              ('imputer', SimpleImputer(strategy = 'mean')),         # Imputation
                                              ('scaler', MinMaxScaler())])                           # Normalization                                        

    Categorical_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')),                  # Imputation
                                                ('encoder', OneHotEncoder(drop = 'first', handle_unknown = 'ignore'))])  # Encoding

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(transformers = [('num', Numerical_transformer, Numerical_variables),
                                                     ('cat', Categorical_transformer, Categorical_variables)])
    return preprocessor
#%% 4. Evaluate 

## ---------------------------- Classifiers ---------------------------------------------------------------
# Model dictionaries (without hyperparameters)
classifiers = {"Logistic Regression": LogisticRegression(random_state = 42, solver = 'saga', class_weight = 'balanced'),
               "Decision Tree": DecisionTreeClassifier(random_state = 42, class_weight = 'balanced'),
               "KNN": KNeighborsClassifier(),
               "SVM": SVC(probability = True, random_state = 42, class_weight = 'balanced')}

# Define parameter grids for classification
classification_param_grids = {"Logistic Regression": {'C': uniform(0.01, 10),
                                                      'penalty': ['l1', 'l2', None],
                                                      'max_iter': [100, 200, 300]},
                              "Decision Tree": {'max_depth': randint(2, 20),
                                                'min_samples_split': randint(2, 20),
                                                'min_samples_leaf': randint(1, 10),
                                                'criterion': ['gini', 'entropy']},
                              "KNN": {'n_neighbors': randint(3, 30),
                                      'weights': ['uniform', 'distance'],
                                      'metric': ['euclidean', 'manhattan', 'minkowski']},
                              "SVM": {'C': uniform(0.1, 10),
                                      'kernel': ['rbf', 'poly', 'sigmoid'],
                                      'gamma': ['scale', 'auto'] + list(uniform(0.001, 1).rvs(5))}}


## ---------------------------- Regressors ---------------------------------------------------------------

regressors = {"OLS": LinearRegression(),
              "Lasso": Lasso(random_state = 42),
              "Ridge": Ridge(random_state = 42),
              "ElasticNet": ElasticNet(random_state = 42),
              "Decision Tree": DecisionTreeRegressor(random_state = 42),
              "KNN": KNeighborsRegressor(),
              "SVM": SVR()}


# Define parameter grids for regression
regression_param_grids = {"OLS": {},  # No hyperparameters to tune
                          "Lasso": {'alpha': uniform(0.001, 2)},
                          "Ridge": {'alpha': uniform(0.001, 2)},
                          "ElasticNet": {'alpha': uniform(0.001, 2), 'l1_ratio': uniform(0, 1)},
                          "Decision Tree": {'max_depth': randint(2, 20),'min_samples_split': randint(2, 20),
                                            'min_samples_leaf': randint(1, 10),
                                            'criterion': ['squared_error', 'friedman_mse', 'absolute_error']},
                          "KNN": {'n_neighbors': randint(3, 30),'weights': ['uniform', 'distance'],
                                  'metric': ['euclidean', 'manhattan', 'minkowski']},
                          "SVM": {'C': uniform(0.1, 10),'kernel': ['rbf', 'poly', 'sigmoid'],
                                  'gamma': ['scale', 'auto']}}



def Evaluate(X_train, X_test, y_train, y_test, preprocessor, classification = True, n_iter = 20, cv_splits = 5,random_state = 42, n_jobs = -1):
    
    # Define metrics and CV strategy
    if classification:
        models_dict = classifiers
        param_grids_dict = classification_param_grids

        metrics = {"Accuracy": "accuracy",
                   "Balanced_Accuracy": "balanced_accuracy",
                   "F1_Score": "f1",
                   "ROC-AUC": "roc_auc",
                   "Precision": "precision",
                   "Sensitivity": "recall"}
        
        cv_strategy = StratifiedKFold(n_splits = cv_splits, shuffle = True, random_state = random_state)
        scoring = "roc_auc"  # Primary metric for tuning
    else:
        models_dict = regressors
        param_grids_dict = regression_param_grids
        
        metrics = {"R2_Score": "r2",
                   "MAE": "neg_mean_absolute_error",
                   "MSE": "neg_mean_squared_error",
                   "RMSE": "neg_root_mean_squared_error"}
        cv_strategy = KFold(n_splits = cv_splits, shuffle = True, random_state = random_state)
        scoring = "neg_mean_absolute_error"  # Primary metric for tuning
    
    Testing_Scores = []
    Validation_Scores = []
    Best_Models = {}
    Tuning_Results = []
    
    for model_name, base_model in models_dict.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        # Create pipeline
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline(steps = [('preprocessor', preprocessor),('model', base_model)])
        
        # Get parameter grid for this model
        param_grid = param_grids_dict.get(model_name, {})
        
        # Add 'model__' prefix to parameters for pipeline
        pipeline_param_grid = {f'model__{key}': value for key, value in param_grid.items()}
        
        # Perform RandomizedSearchCV if parameters are provided
        if pipeline_param_grid:
            print(f"Tuning hyperparameters...")
            random_search = RandomizedSearchCV(estimator = pipeline, param_distributions = pipeline_param_grid,
                                               n_iter = n_iter, cv = cv_strategy, scoring = scoring,
                                               n_jobs = n_jobs, random_state = random_state, verbose = 1)
            random_search.fit(X_train, y_train)
            best_pipeline = random_search.best_estimator_
            
            # Store tuning results
            Tuning_Results.append({"Model": model_name,
                                   "Best_Score": round(random_search.best_score_, 4),
                                   "Best_Params": random_search.best_params_})
            print(f"Best parameters: {random_search.best_params_}")
            print(f"Best CV score: {round(random_search.best_score_, 4)}")
        else:
            print(f"No hyperparameters to tune, using default parameters...")
            best_pipeline = pipeline
            best_pipeline.fit(X_train, y_train)
        
        # Store best model
        Best_Models[model_name] = best_pipeline
        
        # Cross-validation scores on best model
        print(f"Computing cross-validation scores...")
        Evaluation_Metrics = {"Technique": model_name}
        for score_name, scorer in metrics.items():
            scores = cross_val_score(best_pipeline, X_train, y_train, cv = cv_strategy, scoring = scorer, n_jobs = n_jobs)
            mean_score = np.mean(scores)
            if 'neg_' in scorer:
                mean_score = -mean_score
            Evaluation_Metrics[score_name] = round(mean_score, 3)
        Validation_Scores.append(Evaluation_Metrics)
        
        # Test set evaluation
        print(f"Evaluating on test set...")
        predictions = best_pipeline.predict(X_test)
        
        if classification:
            if hasattr(best_pipeline, "predict_proba"):
                prob_predictions = best_pipeline.predict_proba(X_test)[:, 1]
                Test_AUC = round(roc_auc_score(y_test, prob_predictions), 3)
            else:
                Test_AUC = np.nan
            
            Testing_Scores.append({"Technique": model_name,
                                   "Accuracy": round(accuracy_score(y_test, predictions), 3),
                                   "Balanced_Accuracy": round(balanced_accuracy_score(y_test, predictions), 3),
                                   "F1_Score": round(f1_score(y_test, predictions), 3),
                                   "AUC": Test_AUC,
                                   "Precision": round(precision_score(y_test, predictions, zero_division=0), 3),
                                   "Sensitivity": round(recall_score(y_test, predictions), 3)})
        else:
            Testing_Scores.append({"Technique": model_name,
                                   "R2_Score": round(r2_score(y_test, predictions), 3),
                                   "MAE": round(mean_absolute_error(y_test, predictions), 3),
                                   "MSE": round(mean_squared_error(y_test, predictions), 3),
                                   "RMSE": round(root_mean_squared_error(y_test, predictions), 3)})
        
        print(f"{model_name} evaluation complete!")
    
    # Convert to DataFrames
    Validation_Scores_df = pd.DataFrame(Validation_Scores)
    Testing_Scores_df = pd.DataFrame(Testing_Scores)
    Tuning_Results_df = pd.DataFrame(Tuning_Results)
    
    # Save to Excel
    Validation_Scores_df.to_excel('Validation_Scores_df.xlsx', index = False)
    Testing_Scores_df.to_excel('Testing_Scores_df.xlsx', index = False)
    Tuning_Results_df.to_excel('Tuning_Results_df.xlsx', index = False)
    
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")
    
    return Testing_Scores_df, Validation_Scores_df, Best_Models, Tuning_Results_df


#%% 5. Save Best Model
import pickle

def save_best_model(Testing_Scores_df, Best_Models, X, y, classification = True, metric_col = 'AUC'):
    
    # Determine metric to use for selection
    if classification:
        metric_col = metric_col
        ascending = False  # Higher is better
    else:
        metric_col = 'MAE'
        ascending = True   # Lower is better
    
    # Find the best model
    best_row = Testing_Scores_df.sort_values(by = metric_col, ascending = ascending).iloc[0]
    best_model_name = best_row['Technique']
    best_score = best_row[metric_col]
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL SELECTION")
    print(f"{'='*60}")
    print(f"Best Model: {best_model_name}")
    print(f"Best {metric_col}: {best_score}")
    print(f"{'='*60}\n")
    
    # Get the best pipeline and retrain on full data
    final_pipeline = Best_Models[best_model_name]
    final_pipeline.fit(X, y)
    
    # Access the trained model inside the pipeline
    final_model = final_pipeline.named_steps['model']
    
    # Get model-specific information
    model_info = {'model_name': best_model_name,
                  'best_score': best_score,
                  'metric': metric_col}
    
    # Extract model parameters based on model type
    if hasattr(final_model, 'coef_'):
        # Linear models (Logistic Regression, Lasso, Ridge, ElasticNet, OLS)
        model_info['coefficients'] = final_model.coef_
        model_info['intercept'] = final_model.intercept_
        print(f"Coefficients shape: {final_model.coef_.shape}")
        print(f"Intercept: {final_model.intercept_}")
    
    if hasattr(final_model, 'feature_importances_'):
        # Tree-based models (Decision Tree, Random Forest, etc.)
        model_info['feature_importances'] = final_model.feature_importances_
        print(f"Feature importances shape: {final_model.feature_importances_.shape}")
    
    # Save the final pipeline
    with open(f'best_model_{best_model_name.replace(" ", "_")}.pkl', 'wb') as f:
        pickle.dump(final_pipeline, f)
    print(f"\nModel saved as: best_model_{best_model_name.replace(' ', '_')}.pkl")
    
    # Save model info
    with open(f'best_model_info_{best_model_name.replace(" ", "_")}.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    return best_model_name, final_pipeline, model_info
