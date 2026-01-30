import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

# --- SETUP ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Output File
csv_file = 'posture_dataset.csv'

# Check if file exists to write headers
file_exists = os.path.isfile(csv_file)

# Open CSV in Append Mode
f = open(csv_file, 'a', newline='')
writer = csv.writer(f)

# Write Header if new file
if not file_exists:
    headers = ['label'] + [f'v{i}' for i in range(1, 17)]
    writer.writerow(headers)
    print("NEW FILE CREATED: headers written.")

print("--- DATA COLLECTOR ---")
print("Press 'g' to save GOOD posture.")
print("Press 'b' to save BAD posture.")
print("Press 'q' to QUIT.")

cap = cv2.VideoCapture(0)

# Counters
good_count = 0
bad_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Process
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    # Draw
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Collect Data
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        
        # Capture Key
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('g') or key == ord('b'):
            # Extract Nose(0), Ear(7), Shoulder(11), Shoulder(12)
            row = []
            valid_points = True
            for idx in [0, 7, 11, 12]:
                # Safety Check: Ensure landmarks are visible
                if lm[idx].visibility < 0.5:
                    valid_points = False
                row.extend([lm[idx].x, lm[idx].y, lm[idx].z, lm[idx].visibility])
            
            if valid_points:
                label = 'good' if key == ord('g') else 'bad'
                writer.writerow([label] + row)
                
                if label == 'good': good_count += 1
                else: bad_count += 1
                
                print(f"SAVED: {label.upper()} (Total: G={good_count}, B={bad_count})")
            else:
                print("⚠️ POSE NOT CLEAR: Move back so camera sees shoulders!")

    # Display Counts
    cv2.putText(frame, f"Good: {good_count} | Bad: {bad_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Data Collector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
f.close()