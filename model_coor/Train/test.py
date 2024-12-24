# import os
# import pandas as pd
# import joblib
# from sklearn.metrics import accuracy_score, classification_report
# import numpy as np

# # Get the current directory of the script
# current_directory = os.path.dirname(os.path.realpath(__file__))

# # Load the saved model, scaler, and label encoder
# model_filename = os.path.join(current_directory, 'best_pose_classifier_model.joblib')
# scaler_filename = os.path.join(current_directory, 'scaler.joblib')
# label_encoder_filename = os.path.join(current_directory, 'label_encoder.joblib')

# best_model = joblib.load(model_filename)  # Load the trained model
# scaler = joblib.load(scaler_filename)  # Load the scaler
# label_encoder = joblib.load(label_encoder_filename)  # Load the label encoder

# # Load the new test dataset
# test_df = pd.read_csv(os.path.join(current_directory, 'new_test_data.csv'))

# # Separate features and target in the test dataset
# X_test_new = test_df.drop(columns=['Pose Name'])  # Features (all columns except Pose Name)
# y_test_new = test_df['Pose Name']  # Target (Pose Name)

# # Scale the features using the saved scaler
# X_test_scaled = scaler.transform(X_test_new)

# # Encode the actual target labels for comparison
# y_test_encoded = label_encoder.transform(y_test_new)

# # Make predictions with the loaded model
# y_pred_new = best_model.predict(X_test_scaled)

# # Evaluate the accuracy of the model on the new test data
# accuracy = accuracy_score(y_test_encoded, y_pred_new)
# print(f"Accuracy on the new test dataset: {accuracy * 100:.2f}%")

# # Get the correct labels (where predictions match actual labels)
# correct_labels_mask = (y_test_encoded == y_pred_new)

# # Get only the correct labels and their corresponding statistics
# correct_labels = label_encoder.inverse_transform(y_test_encoded[correct_labels_mask])
# correct_pred_labels = label_encoder.inverse_transform(y_pred_new[correct_labels_mask])

# # Optionally, print only the correct label counts
# correct_label_counts = pd.Series(correct_labels).value_counts()

# print("\nCorrect Labels Count:")
# print(correct_label_counts)

# # Optionally, print a classification report but for only the correct labels
# print("\nClassification Report for Correct Labels:")
# print(classification_report(correct_labels, correct_pred_labels))


# import os
# import pandas as pd
# import joblib
# from sklearn.metrics import accuracy_score, classification_report
# import numpy as np

# # Get the current directory of the script
# current_directory = os.path.dirname(os.path.realpath(__file__))

# # Load the saved model, scaler, and label encoder
# model_filename = os.path.join(current_directory, 'best_pose_classifier_model.joblib')
# scaler_filename = os.path.join(current_directory, 'scaler.joblib')
# label_encoder_filename = os.path.join(current_directory, 'label_encoder.joblib')

# best_model = joblib.load(model_filename)  # Load the trained model
# scaler = joblib.load(scaler_filename)  # Load the scaler
# label_encoder = joblib.load(label_encoder_filename)  # Load the label encoder

# # Load the new test dataset
# test_df = pd.read_csv(os.path.join(current_directory, 'new_test_data.csv'))

# # Separate features and target in the test dataset
# X_test_new = test_df.drop(columns=['Pose Name'])  # Features (all columns except Pose Name)
# y_test_new = test_df['Pose Name']  # Target (Pose Name)

# # Scale the features using the saved scaler
# X_test_scaled = scaler.transform(X_test_new)

# # Encode the actual target labels for comparison, handling unseen labels
# y_test_encoded = []
# for label in y_test_new:
#     if label in label_encoder.classes_:
#         y_test_encoded.append(label_encoder.transform([label])[0])  # Encode known labels
#     else:
#         y_test_encoded.append(-1)  # Assign -1 for unseen labels
# y_test_encoded = np.array(y_test_encoded)

# # Make predictions with the loaded model
# y_pred_new = best_model.predict(X_test_scaled)

# # Evaluate the accuracy of the model on the new test data
# accuracy = accuracy_score(y_test_encoded, y_pred_new)
# print(f"Accuracy on the new test dataset: {accuracy * 100:.2f}%")

# # Get the correct labels (where predictions match actual labels)
# correct_labels_mask = (y_test_encoded == y_pred_new)

# # Get only the correct labels and their corresponding statistics
# correct_labels = label_encoder.inverse_transform(y_test_encoded[correct_labels_mask]) if y_test_encoded[correct_labels_mask].size > 0 else []
# correct_pred_labels = label_encoder.inverse_transform(y_pred_new[correct_labels_mask]) if y_pred_new[correct_labels_mask].size > 0 else []

# # Optionally, print only the correct label counts
# correct_label_counts = pd.Series(correct_labels).value_counts()

# print("\nCorrect Labels Count:")
# print(correct_label_counts)

# # Optionally, print a classification report but for only the correct labels
# print("\nClassification Report for Correct Labels:")
# if correct_labels:
#     print(classification_report(correct_labels, correct_pred_labels))
# else:
#     print("No correct labels found.")


import os
import joblib
import numpy as np

# Get the current directory of the script
current_directory = os.path.dirname(os.path.realpath(__file__))

# Load the saved model, scaler, and label encoder
model_filename = os.path.join(current_directory, 'best_pose_classifier_model.joblib')
scaler_filename = os.path.join(current_directory, 'scaler.joblib')
label_encoder_filename = os.path.join(current_directory, 'label_encoder.joblib')

best_model = joblib.load(model_filename)  # Load the trained model
scaler = joblib.load(scaler_filename)  # Load the scaler
label_encoder = joblib.load(label_encoder_filename)  # Load the label encoder

# Prepare the new input data
new_input_data = np.array([[178.66065697580464, 175.0743424736985, 84.18391204798971, 174.24259943271952,
                            178.13476712394598, 87.42300938123034, 118.21021502595842, 101.63950949264904]])

# Scale the features using the saved scaler
new_input_scaled = scaler.transform(new_input_data)

# Make predictions with the loaded model
y_pred_new = best_model.predict(new_input_scaled)

# Decode the predicted label
predicted_label = label_encoder.inverse_transform(y_pred_new)

# Output the predicted label
print("Predicted Label:")
print(predicted_label[0])  
