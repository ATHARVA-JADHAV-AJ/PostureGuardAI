import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
import pandas as pd
import ctypes
import os

# --- 1. SETUP & IMPORTS ---
try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except ImportError:
    import mediapipe.solutions.pose as mp_pose
    import mediapipe.solutions.drawing_utils as mp_drawing

# --- 2. THE VISUAL MONITOR CLASS ---
class PostureMonitorAI:
    def __init__(self):
        print("--- INITIALIZING VISUAL DASHBOARD ---")
        
        # A. Load the Brain (Strict Mode)
        model_path = 'posture_model.pkl'
        self.model = None
        
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print(f"SUCCESS: Loaded {model_path} 🧠")
            except Exception as e:
                print(f"ERROR: Could not load model. Reason: {e}")
        else:
            print(f"CRITICAL WARNING: '{model_path}' not found!")
            print("Make sure you ran 'train_model.py' first!")

        # B. Setup Eyes
        self.mp_pose = mp_pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp_drawing
        
        # C. Sensitivity Settings
        self.STAGNATION_LIMIT = 50      
        self.SLOUCH_LIMIT = 3           # Faster reaction for AI
        self.MOVE_THRESHOLD = 0.008     
        
        self.LONG_BREAK_MINUTES = 45    
        self.session_start_time = time.time()
        
        # D. State & Visuals
        self.prev_keypoints = None
        self.last_move_time = time.time()
        self.first_bad_posture_time = None
        self.last_notify_time = 0
        self.current_move_intensity = 0 

    def send_notification(self, title, message):
        """Forces a visible popup window."""
        try:
            ctypes.windll.user32.MessageBoxW(0, message, title, 0x40000)
        except:
            print(f"ALERT: {title} - {message}")

    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        curr_time = time.time()
        
        # Default Visuals
        status_text = "Status: No Human"
        status_color = (100, 100, 100) # Gray

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            status_text = "Analyzing..." 
            status_color = (255, 200, 0) # Amber

            # --- LOGIC 1: AI POSTURE CHECK (STRICT) ---
            is_slouching = False
            
            if self.model is None:
                status_text = "ERR: NO MODEL FILE"
                status_color = (0, 0, 255)
            else:
                # Extract EXACT features expected by the model
                # (Nose, Ear L, Shoulder L, Shoulder R)
                row = []
                for idx in [0, 7, 11, 12]:
                    row.extend([lm[idx].x, lm[idx].y, lm[idx].z, lm[idx].visibility])
                
                try:
                    # Convert to DataFrame to match training format
                    feature_names = [f'v{i}' for i in range(1, 17)]
                    X_input = pd.DataFrame([row], columns=feature_names)
                    
                    # ASK THE AI
                    prediction = self.model.predict(X_input)[0]
                    
                    if prediction == 'bad': 
                        is_slouching = True
                    else:
                        is_slouching = False
                        
                except Exception as e:
                    print(f"Prediction Error: {e}")

            # Apply Result
            if is_slouching:
                status_text = "AI: BAD POSTURE"
                status_color = (0, 0, 255) # Red
                
                if self.first_bad_posture_time is None:
                    self.first_bad_posture_time = curr_time
                elif (curr_time - self.first_bad_posture_time > self.SLOUCH_LIMIT):
                    if (curr_time - self.last_notify_time > 60):
                        self.send_notification("Posture Check", "Sit up straight!")
                        self.last_notify_time = curr_time
            elif self.model is not None:
                status_text = "AI: GOOD POSTURE"
                status_color = (0, 255, 0) # Green
                self.first_bad_posture_time = None 

            # --- LOGIC 2: STAGNATION & TIMERS ---
            # Session Timer
            session_duration = curr_time - self.session_start_time
            if session_duration > (self.LONG_BREAK_MINUTES * 60):
                self.send_notification("HEALTH ALERT", "Time to take a walk!")
                self.session_start_time = curr_time 

            # Movement Check
            curr_p = np.array([[lm[0].x, lm[0].y], [lm[11].x, lm[11].y], [lm[12].x, lm[12].y]])
            if self.prev_keypoints is not None:
                movements = np.linalg.norm(curr_p - self.prev_keypoints, axis=1)
                self.current_move_intensity = np.max(movements) 
                if self.current_move_intensity > self.MOVE_THRESHOLD:
                    self.last_move_time = curr_time
            self.prev_keypoints = curr_p
            
            stagnation_time = curr_time - self.last_move_time
            if stagnation_time > self.STAGNATION_LIMIT and (curr_time - self.last_notify_time > 60):
                self.send_notification("Movement Alert", "Move your body!")
                self.last_notify_time = curr_time

            # --- VISUALS ---
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Status Box
            cv2.rectangle(frame, (5, 5), (350, 60), (0,0,0), -1)
            cv2.putText(frame, status_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # Movement Bar
            bar_width = int(min(self.current_move_intensity * 5000, 200))
            cv2.rectangle(frame, (10, 80), (10 + bar_width, 95), (255, 255, 0), -1) 
            cv2.rectangle(frame, (10, 80), (210, 95), (255, 255, 255), 1) 
            cv2.putText(frame, "Movement Intensity", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Timers
            mins_sat = int(session_duration // 60)
            cv2.putText(frame, f"Session: {mins_sat}m / {self.LONG_BREAK_MINUTES}m", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            timer_color = (0, 255, 255)
            if stagnation_time > 40: timer_color = (0, 0, 255)   
            cv2.putText(frame, f"Stillness: {stagnation_time:.1f}s", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_color, 2)

        return frame

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    monitor = PostureMonitorAI()
    
    print("--- POSTURE GOAT: AI VERSION ---")
    print("Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        output = monitor.process_frame(frame)
        cv2.imshow('Posture GOAT', output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()