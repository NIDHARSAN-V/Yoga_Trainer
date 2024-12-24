# import cv2
# import mediapipe as mp


# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils

# video_path = "./video.mp4"
# cap = cv2.VideoCapture(video_path)

# # Initialize Pose model
# with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("End of video.")
#             break

#         frame = cv2.resize(frame, (1280, 720))

#         # Convert the image to RGB
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False

#         # Detect pose landmarks
#         results = pose.process(image)

#         # Convert the image back to BGR for OpenCV
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         # Draw pose landmarks with additional styling
#         if results.pose_landmarks:
#             landmarks = results.pose_landmarks.landmark

#             # Print specific body part coordinates
#             # Neck (roughly between the shoulders)
#             neck = landmarks[1]  # LEFT_SHOULDER
#             print(f"Neck (LEFT_SHOULDER) coordinates: (x: {neck.x}, y: {neck.y}, z: {neck.z})")

#             # Knees
#             left_knee = landmarks[25]
#             right_knee = landmarks[26]
#             print(f"Left Knee coordinates: (x: {left_knee.x}, y: {left_knee.y}, z: {left_knee.z})")
#             print(f"Right Knee coordinates: (x: {right_knee.x}, y: {right_knee.y}, z: {right_knee.z})")

#             # Hips
#             left_hip = landmarks[23]
#             right_hip = landmarks[24]
#             print(f"Left Hip coordinates: (x: {left_hip.x}, y: {left_hip.y}, z: {left_hip.z})")
#             print(f"Right Hip coordinates: (x: {right_hip.x}, y: {right_hip.y}, z: {right_hip.z})")

#             # Legs (Knee to Ankle)
#             left_ankle = landmarks[27]
#             right_ankle = landmarks[28]
#             print(f"Left Leg (Left Knee to Left Ankle) coordinates: (x: {left_knee.x}, y: {left_knee.y}, z: {left_knee.z}) -> (x: {left_ankle.x}, y: {left_ankle.y}, z: {left_ankle.z})")
#             print(f"Right Leg (Right Knee to Right Ankle) coordinates: (x: {right_knee.x}, y: {right_knee.y}, z: {right_knee.z}) -> (x: {right_ankle.x}, y: {right_ankle.y}, z: {right_ankle.z})")

#             # Hands
#             left_hand = landmarks[15]  # LEFT_WRIST
#             right_hand = landmarks[16]  # RIGHT_WRIST
#             print(f"Left Hand (LEFT_WRIST) coordinates: (x: {left_hand.x}, y: {left_hand.y}, z: {left_hand.z})")
#             print(f"Right Hand (RIGHT_WRIST) coordinates: (x: {right_hand.x}, y: {right_hand.y}, z: {right_hand.z})")

#             # Draw pose landmarks
#             mp_drawing.draw_landmarks(
#                 image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                 mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
#                 mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
#             )

#         # Display the frame with landmarks
#         cv2.imshow("Pose Landmarks", image)

#         # Break the loop on 'q' key press
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

# # Release the video capture object and close OpenCV windows
# cap.release()
# cv2.destroyAllWindows()



# /////////////////////////////////////////////////////////////////////////////////////////


import cv2
import mediapipe as mp
import numpy as np

# Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Video path
video_path = "./video.mp4"
cap = cv2.VideoCapture(video_path)

# Reference angles/coordinates for scoring
reference_angles = {
    "neck_alignment": 90,  # Ideal neck angle
    "hip_symmetry": 0,     # Hips should be level
    "knee_alignment": 180, # Ideal straight leg alignment
    "elbow_alignment": 180, # Straight arm alignment
    "wrist_elbow_alignment": 90 # Right angle between wrist and elbow
}

# Function to calculate the angle between three points
def calculate_angle(p1, p2, p3):
    a = np.array([p1.x, p1.y])  # Point 1
    b = np.array([p2.x, p2.y])  # Point 2 (vertex)
    c = np.array([p3.x, p3.y])  # Point 3

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle if angle <= 180 else 360 - angle

# Initialize Pose model
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        frame = cv2.resize(frame, (1280, 720))

        # Convert the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Detect pose landmarks
        results = pose.process(image)

        # Convert the image back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Define landmarks for calculations
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Calculate angles for scoring
            neck_alignment = calculate_angle(left_shoulder, right_shoulder, landmarks[mp_pose.PoseLandmark.NOSE])
            hip_symmetry = abs(left_hip.y - right_hip.y)
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            left_wrist_elbow_angle = calculate_angle(left_elbow, left_wrist, landmarks[mp_pose.PoseLandmark.LEFT_THUMB])
            right_wrist_elbow_angle = calculate_angle(right_elbow, right_wrist, landmarks[mp_pose.PoseLandmark.RIGHT_THUMB])

            # Scoring
            scores = {
                "neck_alignment": max(0, 100 - abs(neck_alignment - reference_angles["neck_alignment"])),
                "hip_symmetry": max(0, 100 - (hip_symmetry * 100)),
                "left_knee_alignment": max(0, 100 - abs(left_knee_angle - reference_angles["knee_alignment"])),
                "right_knee_alignment": max(0, 100 - abs(right_knee_angle - reference_angles["knee_alignment"])),
                "left_elbow_alignment": max(0, 100 - abs(left_elbow_angle - reference_angles["elbow_alignment"])),
                "right_elbow_alignment": max(0, 100 - abs(right_elbow_angle - reference_angles["elbow_alignment"])),
                "left_wrist_elbow_alignment": max(0, 100 - abs(left_wrist_elbow_angle - reference_angles["wrist_elbow_alignment"])),
                "right_wrist_elbow_alignment": max(0, 100 - abs(right_wrist_elbow_angle - reference_angles["wrist_elbow_alignment"])),
            }

            # Print scores
            print(f"Scores:")
            for key, value in scores.items():
                print(f" - {key}: {value:.2f}")

            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # Display the frame with landmarks
        cv2.imshow("Pose Landmarks", image)

        # Break the loop on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the video capture object and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
