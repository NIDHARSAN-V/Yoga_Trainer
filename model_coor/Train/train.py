# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from imblearn.over_sampling import SMOTE
# import joblib
# import xgboost as xgb

# # Get the current directory of the script
# current_directory = os.path.dirname(os.path.realpath(__file__))

# # Load the dataset from CSV file
# df = pd.read_csv(os.path.join(current_directory, 'final_data.csv'))

# # Separate features and target
# X = df.drop(columns=['Pose Name'])  # Features (all columns except Pose Name)
# y = df['Pose Name']  # Target (Pose Name)

# # Encode categorical target labels into numeric values
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# # Scale the features (Standardize the data)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Apply SMOTE for handling class imbalance
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)

# # Split the resampled data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# # Hyperparameter tuning with GridSearchCV
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 20, 30, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
#                            param_grid=param_grid, 
#                            cv=5, 
#                            n_jobs=-1, 
#                            verbose=2)
# grid_search.fit(X_train, y_train)

# # Best model after GridSearchCV
# best_model = grid_search.best_estimator_

# # Evaluate the best model
# y_pred = best_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy after hyperparameter tuning: {accuracy * 100:.2f}%")

# # Perform cross-validation using the best model from GridSearchCV
# cross_val_scores_best_model = cross_val_score(best_model, X_resampled, y_resampled, cv=5, n_jobs=-1)
# print(f"Cross-validation scores (best model): {cross_val_scores_best_model}")
# print(f"Average cross-validation score (best model): {cross_val_scores_best_model.mean() * 100:.2f}%")

# # Save the trained model
# model_filename = os.path.join(current_directory, 'pose_classifier_model.joblib')
# joblib.dump(best_model, model_filename)

# # Load the saved model and make predictions
# loaded_model = joblib.load(model_filename)
# y_pred_loaded_model = loaded_model.predict(X_test)

# # Calculate accuracy with the loaded model
# accuracy_loaded_model = accuracy_score(y_test, y_pred_loaded_model)
# print(f'Accuracy of the loaded model: {accuracy_loaded_model * 100:.2f}%')

# # Optional: Using XGBoost for potentially better performance
# xgb_model = xgb.XGBClassifier(random_state=42)
# xgb_model.fit(X_train, y_train)

# # Make predictions and calculate accuracy for XGBoost
# y_pred_xgb = xgb_model.predict(X_test)
# accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
# print(f'Accuracy of the XGBoost model: {accuracy_xgb * 100:.2f}%')



import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

# Get the current directory of the script
current_directory = os.path.dirname(os.path.realpath(__file__))

# Load the dataset from CSV file
df = pd.read_csv(os.path.join(current_directory, 'final_data.csv'))

# Separate features and target
X = df.drop(columns=['Pose Name'])  # Features (all columns except Pose Name)
y = df['Pose Name']  # Target (Pose Name)

# Encode categorical target labels into numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale the features (Standardize the data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE for handling class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                           param_grid=param_grid, 
                           cv=5, 
                           n_jobs=-1, 
                           verbose=2)
grid_search.fit(X_train, y_train)

# Best model after GridSearchCV
best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy after hyperparameter tuning: {accuracy * 100:.2f}%")

# Save the trained model, scaler, and label encoder
model_filename = os.path.join(current_directory, 'best_pose_classifier_model.joblib')
scaler_filename = os.path.join(current_directory, 'scaler.joblib')
label_encoder_filename = os.path.join(current_directory, 'label_encoder.joblib')

# Save the model, scaler, and label encoder
joblib.dump(best_model, model_filename)  # Save the model
joblib.dump(scaler, scaler_filename)  # Save the scaler
joblib.dump(label_encoder, label_encoder_filename)  # Save the label encoder

print(f"Best model, Scaler, and Label Encoder saved successfully.")

# Optionally, you can also check the grid search results for the best parameters:
print(f"Best hyperparameters: {grid_search.best_params_}")
