import cv2
import mediapipe as mp

# Initialize Mediapipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load the image file
image_path = "./pose.jpg"  # Replace with the path to your image
image = cv2.imread(image_path)

# Pose detection setup with higher confidence for better accuracy
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    # Convert the image to RGB (Mediapipe requires RGB format)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image.flags.writeable = False

    # Detect pose landmarks
    results = pose.process(rgb_image)

    # Convert the image back to BGR for OpenCV
    rgb_image.flags.writeable = True
    image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # Draw pose landmarks and print the coordinates
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Print specific body part coordinates
        # Neck (roughly between the shoulders)
        neck = landmarks[1]  # LEFT_SHOULDER
        print(f"Neck (LEFT_SHOULDER) coordinates: (x: {neck.x}, y: {neck.y}, z: {neck.z})")

        # Knees
        left_knee = landmarks[25]
        right_knee = landmarks[26]
        print(f"Left Knee coordinates: (x: {left_knee.x}, y: {left_knee.y}, z: {left_knee.z})")
        print(f"Right Knee coordinates: (x: {right_knee.x}, y: {right_knee.y}, z: {right_knee.z})")

        # Hips
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        print(f"Left Hip coordinates: (x: {left_hip.x}, y: {left_hip.y}, z: {left_hip.z})")
        print(f"Right Hip coordinates: (x: {right_hip.x}, y: {right_hip.y}, z: {right_hip.z})")

        # Legs (Knee to Ankle)
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        print(f"Left Leg (Left Knee to Left Ankle) coordinates: (x: {left_knee.x}, y: {left_knee.y}, z: {left_knee.z}) -> (x: {left_ankle.x}, y: {left_ankle.y}, z: {left_ankle.z})")
        print(f"Right Leg (Right Knee to Right Ankle) coordinates: (x: {right_knee.x}, y: {right_knee.y}, z: {right_knee.z}) -> (x: {right_ankle.x}, y: {right_ankle.y}, z: {right_ankle.z})")

        # Hands
        left_hand = landmarks[15]  # LEFT_WRIST
        right_hand = landmarks[16]  # RIGHT_WRIST
        print(f"Left Hand (LEFT_WRIST) coordinates: (x: {left_hand.x}, y: {left_hand.y}, z: {left_hand.z})")
        print(f"Right Hand (RIGHT_WRIST) coordinates: (x: {right_hand.x}, y: {right_hand.y}, z: {right_hand.z})")

        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

    # Display the image with landmarks
    cv2.imshow("Pose Landmarks", image)

    # Wait until a key is pressed to close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
