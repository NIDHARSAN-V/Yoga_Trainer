import os
import base64
import cv2
import numpy as np
import joblib
import math
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import mediapipe as mp

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Load models, scaler, and label encoder
current_directory = os.path.dirname(os.path.realpath(__file__))
model_filename = os.path.join(current_directory, 'best_pose_classifier_model.joblib')
scaler_filename = os.path.join(current_directory, 'scaler.joblib')
label_encoder_filename = os.path.join(current_directory, 'label_encoder.joblib')

best_model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)
label_encoder = joblib.load(label_encoder_filename)

# Function to predict pose
def predict_pose(features):
    scaled_features = scaler.transform([features])
    prediction = best_model.predict(scaled_features)
    label = label_encoder.inverse_transform(prediction)
    return label[0]

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    ba_x = a.x - b.x
    ba_y = a.y - b.y
    bc_x = c.x - b.x
    bc_y = c.y - b.y

    dot_product = (ba_x * bc_x) + (ba_y * bc_y)
    magnitude_ba = math.sqrt(ba_x**2 + ba_y**2)
    magnitude_bc = math.sqrt(bc_x**2 + bc_y**2)

    if magnitude_ba * magnitude_bc == 0:
        return 0

    angle = math.acos(dot_product / (magnitude_ba * magnitude_bc))
    return math.degrees(angle)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

@app.route('/')
def index():
    return render_template('index.html')

# SocketIO event to handle the image from frontend
@socketio.on('image')
def handle_image(data):
    try:
        # Decode base64 image
        img_data = base64.b64decode(data)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            emit('error', {'error': 'Invalid frame data'})
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe Pose
        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Calculate angles
                features = [
                    calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
                                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST]),
                    calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
                                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]),
                    calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE]),
                    calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]),
                    calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]),
                    calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]),
                    calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP]),
                    calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP]),
                ]

                # Predict pose label
                label = predict_pose(features)

                # Send prediction and processed frame back to client
                _, buffer = cv2.imencode('.jpg', frame)
                encoded_frame = base64.b64encode(buffer).decode('utf-8')

                emit('prediction', {'label': label, 'frame': encoded_frame})
            else:
                emit('error', {'error': 'No landmarks detected'})
    except Exception as e:
        emit('error', {'error': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True)
