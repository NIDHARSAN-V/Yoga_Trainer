import cv2
import mediapipe as mp
import csv
import os

# Initialize Mediapipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Path to the folder containing images
folder_path = "./Vrksasana"  # Replace with the correct path to your folder

# Set up CSV file for storing the coordinates
csv_file = "pose_landmarks.csv"

# Write header to CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Header for CSV file
    writer.writerow([
        "Left Knee X", "Left Knee Y", "Left Knee Z",
        "Right Knee X", "Right Knee Y", "Right Knee Z",
        "Left Hip X", "Left Hip Y", "Left Hip Z",
        "Right Hip X", "Right Hip Y", "Right Hip Z",
        "Left Leg X", "Left Leg Y", "Left Leg Z",  # Knee to Ankle
        "Right Leg X", "Right Leg Y", "Right Leg Z",  # Knee to Ankle
        "Left Hand X", "Left Hand Y", "Left Hand Z",
        "Right Hand X", "Right Hand Y", "Right Hand Z",
        "Label"
    ])

# Iterate over each image in the folder
for image_name in os.listdir(folder_path):
    image_path = os.path.join(folder_path, image_name)

    # Check if it's a valid image file (you can filter by file extension)
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter image files
        image = cv2.imread(image_path)

        # Check if the image is loaded properly
        if image is None:
            print(f"Error: Unable to load image at {image_path}")
            continue  # Skip this image and move to the next one

        # Get the folder name as the label
        label = os.path.basename(folder_path)

        # Pose detection setup with higher confidence for better accuracy
        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            # Convert the image to RGB (Mediapipe requires RGB format)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb_image.flags.writeable = False

            # Detect pose landmarks
            results = pose.process(rgb_image)

            # Collect the coordinates of the body parts for the current image
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Coordinates for the body parts (Left Knee, Right Knee, Left Hip, Right Hip, etc.)
                left_knee = landmarks[25]
                right_knee = landmarks[26]
                left_hip = landmarks[23]
                right_hip = landmarks[24]
                left_ankle = landmarks[27]
                right_ankle = landmarks[28]
                left_hand = landmarks[15]
                right_hand = landmarks[16]

                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        left_knee.x, left_knee.y, left_knee.z,
                        right_knee.x, right_knee.y, right_knee.z,
                        left_hip.x, left_hip.y, left_hip.z,
                        right_hip.x, right_hip.y, right_hip.z,
                        left_ankle.x, left_ankle.y, left_ankle.z,
                        right_ankle.x, right_ankle.y, right_ankle.z,
                        left_hand.x, left_hand.y, left_hand.z,
                        right_hand.x, right_hand.y, right_hand.z,
                        label  
                    ])

cv2.destroyAllWindows() 
