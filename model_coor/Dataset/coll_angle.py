import cv2
import mediapipe as mp
import math
import os
import csv

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

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Directory containing images (datas folder)
image_directory = "./datas"
output_csv = "./csv/pose_angles.csv"

# Prepare CSV file for writing
csv_headers = [
    "Left Elbow Angle", "Left Knee Angle", "Left Hip Angle", 
    "Right Elbow Angle", "Right Knee Angle", "Right Hip Angle",
    "Left Shoulder Angle", "Right Shoulder Angle", "Pose Name"
]

# Open CSV file in write mode
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_headers)

    # Iterate over each folder in the 'datas' directory
    for pose_folder in os.listdir(image_directory):
        pose_folder_path = os.path.join(image_directory, pose_folder)
        
        if os.path.isdir(pose_folder_path):  # Process only folders
            # Iterate over all image files in the folder
            for image_name in os.listdir(pose_folder_path):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
                    image_path = os.path.join(pose_folder_path, image_name)
                    
                    # Read and process the image
                    image = cv2.imread(image_path)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
                        results = pose.process(image_rgb)

                        if results.pose_landmarks:
                            # Access landmarks and calculate angles
                            landmarks = results.pose_landmarks.landmark

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

                            # Add shoulder angles
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

                            # Write results to CSV with the folder name as pose name
                            writer.writerow([ 
                                left_elbow_angle, left_knee_angle, left_hip_angle, 
                                right_elbow_angle, right_knee_angle, right_hip_angle,
                                left_shoulder_angle, right_shoulder_angle, pose_folder.lower()
                            ])

                            # Optional: Visualize landmarks and angles
                            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                            # Annotating angles on the image
                            cv2.putText(image, f'L Elbow: {int(left_elbow_angle)}°', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                            cv2.putText(image, f'R Elbow: {int(right_elbow_angle)}°', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                            cv2.putText(image, f'L Shoulder: {int(left_shoulder_angle)}°', (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                            cv2.putText(image, f'R Shoulder: {int(right_shoulder_angle)}°', (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                            # Display processed image with annotations
                            cv2.imshow(f"Pose Estimation - {image_name}", image)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                        else:
                            print(f"No pose landmarks detected in image: {image_name}")
