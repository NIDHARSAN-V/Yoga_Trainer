import os
import base64
import cv2
import numpy as np
import joblib
import math
from flask import Flask, render_template, request, jsonify
import mediapipe as mp

app = Flask(__name__)

def predict(left_elbow_angle, left_knee_angle, left_hip_angle, right_elbow_angle,
            right_knee_angle, right_hip_angle, left_shoulder_angle, right_shoulder_angle):
    current_directory = os.path.dirname(os.path.realpath(__file__))

    # Load the saved model, scaler, and label encoder
    model_filename = os.path.join(current_directory, 'best_pose_classifier_model.joblib')
    scaler_filename = os.path.join(current_directory, 'scaler.joblib')
    label_encoder_filename = os.path.join(current_directory, 'label_encoder.joblib')

    best_model = joblib.load(model_filename)  # Load the trained model
    scaler = joblib.load(scaler_filename)  # Load the scaler
    label_encoder = joblib.load(label_encoder_filename)  # Load the label encoder

    # Prepare the new input data
    new_input_data = np.array([[left_elbow_angle, left_knee_angle, left_hip_angle, right_elbow_angle,
                                right_knee_angle, right_hip_angle, left_shoulder_angle, right_shoulder_angle]])

    # Scale the features using the saved scaler
    new_input_scaled = scaler.transform(new_input_data)

    # Make predictions with the loaded model
    y_pred_new = best_model.predict(new_input_scaled)

    # Decode the predicted label
    predicted_label = label_encoder.inverse_transform(y_pred_new)

    return predicted_label[0]

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    """
    Calculate the angle between three points
    :param a: First point (landmark) as a MediaPipe landmark
    :param b: Middle point (landmark) as a MediaPipe landmark
    :param c: Last point (landmark) as a MediaPipe landmark
    :return: Angle in degrees
    """
    # Vector BA
    ba_x = a.x - b.x
    ba_y = a.y - b.y

    # Vector BC
    bc_x = c.x - b.x
    bc_y = c.y - b.y

    # Dot product and magnitudes
    dot_product = (ba_x * bc_x) + (ba_y * bc_y)
    magnitude_ba = math.sqrt(ba_x**2 + ba_y**2)
    magnitude_bc = math.sqrt(bc_x**2 + bc_y**2)

    # Handle divide by zero
    if magnitude_ba * magnitude_bc == 0:
        return 0

    # Calculate angle in radians and convert to degrees
    angle = math.acos(dot_product / (magnitude_ba * magnitude_bc))
    return math.degrees(angle)

# Function to validate presence of all required landmarks
def validate_landmarks(landmarks, required_landmarks):
    """
    Check if all required landmarks are present in the detected landmarks.
    :param landmarks: List of detected pose landmarks
    :param required_landmarks: List of PoseLandmark enums that must be present
    :return: Boolean indicating whether all required landmarks are present
    """
    return all(landmarks[landmark].visibility > 0.5 for landmark in required_landmarks)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_pose():
    # Get the base64 image from the form
    image_data = request.form['image']

    # Ensure the base64 string is valid
    if ',' in image_data:
        image_data = image_data.split(',')[1]  # Extract the base64 data
    else:
        return jsonify({'error': 'Invalid base64 image data.'})

    try:
        # Decode the base64 string
        img_data = base64.b64decode(image_data)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'})

    if image is None:
        return jsonify({'error': 'Unable to process image.'})

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Define required landmarks
            required_landmarks = [
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.LEFT_WRIST,
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.LEFT_ANKLE,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_WRIST,
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.RIGHT_ANKLE,
            ]

            # Validate landmarks
            if not validate_landmarks(landmarks, required_landmarks):
                return jsonify({'error': 'Full body is not visible in the image. Please provide an image where the full body is clearly visible.'})

            # Calculate angles
            left_elbow_angle = calculate_angle(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            )
            left_knee_angle = calculate_angle(
                landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            )
            left_hip_angle = calculate_angle(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            )

            right_elbow_angle = calculate_angle(
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            )
            right_knee_angle = calculate_angle(
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            )
            right_hip_angle = calculate_angle(
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            )

            left_shoulder_angle = calculate_angle(
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            )
            right_shoulder_angle = calculate_angle(
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            )

            # Predict the label
            predicted_label = predict(left_elbow_angle, left_knee_angle, left_hip_angle, right_elbow_angle,
                                      right_knee_angle, right_hip_angle, left_shoulder_angle, right_shoulder_angle)

            return jsonify({'label': predicted_label})
        else:
            return jsonify({'error': 'No pose landmarks detected in the image.'})


if __name__ == '__main__':
    app.run(debug=True)
