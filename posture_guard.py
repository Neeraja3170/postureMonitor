import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import json
import os
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import platform

# Initialize pygame for sound
pygame.mixer.init()
try:
    alert_sound = pygame.mixer.Sound('alert.wav')
    success_sound = pygame.mixer.Sound('success.wav')
except:
    alert_sound = None
    success_sound = None

# Platform-specific beep function
def play_beep():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 500)  # Frequency: 1000Hz, Duration: 500ms
    else:
        # For Mac and Linux
        print('\a')  # System bell
        os.system('echo -e "\a"')  # Alternative method

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class PostureGuardAI:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Posture tracking variables
        self.bad_posture_counter = 0
        self.bad_posture_threshold = 15
        self.good_posture_time = 0
        self.last_alert_time = 0
        self.alert_cooldown = 5
        self.angle_history = deque(maxlen=5)
        
        # Session tracking
        self.session_start = datetime.now()
        self.posture_data = {
            "good_posture_time": 0,
            "bad_posture_time": 0,
            "posture_changes": 0,
            "posture_history": []
        }
        
        # Gamification elements
        self.streak_days = 0
        self.last_session_date = None
        self.points = 0
        self.load_user_data()
        
        # Posture correction coach
        self.correction_tips = [
            "Adjust your chair height so your feet are flat on the floor",
            "Keep your screen at eye level to avoid neck strain",
            "Relax your shoulders, don't let they creep up toward your ears",
            "Keep your elbows close to your body at a 90-120 degree angle",
            "Take a 2-minute break every 30 minutes to stretch"
        ]
        self.current_tip = 0
        self.last_tip_time = time.time()
        self.tip_interval = 30  # seconds
        
        # AI personalization
        self.user_posture_baseline = None
        self.adaptive_threshold = 140  # Starting threshold
        
        # Initialize pose estimation
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=2  # Use more complex model for better accuracy
        )
        
        print("Starting Advanced PostureGuard AI... Press 'q' to quit.")
        print("Unique features: AI personalization, gamification, progress tracking, and more!")
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def calculate_distance(self, a, b):
        """Calculate Euclidean distance between two points."""
        a = np.array(a)
        b = np.array(b)
        return np.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))

    def play_alert(self):
        """Play alert sound."""
        # Use the simple beep function for wrong postures
        play_beep()

    def play_success(self):
        """Play success sound."""
        if success_sound:
            success_sound.play()

    def load_user_data(self):
        """Load user data from JSON file."""
        try:
            if os.path.exists("posture_data.json"):
                with open("posture_data.json", "r") as f:
                    data = json.load(f)
                    self.streak_days = data.get("streak_days", 0)
                    self.last_session_date = data.get("last_session_date")
                    self.points = data.get("points", 0)
                    self.user_posture_baseline = data.get("posture_baseline")
                    
                    # Check if we need to update streak
                    if self.last_session_date:
                        last_date = datetime.strptime(self.last_session_date, "%Y-%m-%d")
                        days_since = (datetime.now() - last_date).days
                        if days_since == 1:  # Used yesterday
                            self.streak_days += 1
                        elif days_since > 1:  # Broken streak
                            self.streak_days = 1
        except:
            self.streak_days = 0
            self.points = 0

    def save_user_data(self):
        """Save user data to JSON file."""
        data = {
            "streak_days": self.streak_days,
            "last_session_date": datetime.now().strftime("%Y-%m-%d"),
            "points": self.points,
            "posture_baseline": self.user_posture_baseline
        }
        
        with open("posture_data.json", "w") as f:
            json.dump(data, f)

    def generate_progress_chart(self):
        """Generate a progress chart for the current session."""
        if len(self.posture_data["posture_history"]) < 2:
            return None
            
        fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
        times = [t[0] for t in self.posture_data["posture_history"]]
        angles = [t[1] for t in self.posture_data["posture_history"]]
        
        ax.plot(times, angles, 'b-', label='Neck Angle')
        ax.axhline(y=self.adaptive_threshold, color='r', linestyle='--', label='Good Posture Threshold')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Neck Angle (degrees)')
        ax.set_title('Posture During Session')
        ax.legend()
        ax.grid(True)
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        chart = np.asarray(buf)
        chart = cv2.cvtColor(chart, cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        
        return chart

    def detect_posture(self, landmarks, image_shape):
        """Detect different types of poor posture with enhanced logic."""
        height, width = image_shape[:2]
        
        try:
            # Get key landmarks
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x * width,
                    landmarks[mp_pose.PoseLandmark.NOSE.value].y * height]
            
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
            
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height]
            
            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * width,
                        landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * height]
            
            right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * width,
                         landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y * height]
            
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
            
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * width,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * height]
            
            # Calculate midpoints
            shoulder_mid = [(left_shoulder[0] + right_shoulder[0]) / 2, 
                            (left_shoulder[1] + right_shoulder[1]) / 2]
            
            hip_mid = [(left_hip[0] + right_hip[0]) / 2, 
                       (left_hip[1] + right_hip[1]) / 2]
            
            ear_mid = [(left_ear[0] + right_ear[0]) / 2, 
                       (left_ear[1] + right_ear[1]) / 2]
            
            # Calculate angles for posture assessment
            neck_angle = self.calculate_angle(ear_mid, shoulder_mid, hip_mid)
            
            # Calculate shoulder tilt (horizontal alignment)
            shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1])
            
            # Calculate body lean (vertical alignment)
            body_lean = abs(shoulder_mid[0] - hip_mid[0])
            
            # Calculate if person is lying down (based on vertical position of shoulders)
            vertical_position = shoulder_mid[1] / height
            
            # Adaptive threshold - personalizes based on user's baseline
            if self.user_posture_baseline:
                threshold = self.user_posture_baseline * 0.95  # 5% allowance
            else:
                threshold = self.adaptive_threshold
            
            # Posture detection logic
            if vertical_position > 0.7:
                return "lying", neck_angle, threshold
            
            elif neck_angle < threshold:
                return "slouching", neck_angle, threshold
            
            elif body_lean > 50:
                return "leaning", neck_angle, threshold
            
            elif shoulder_tilt > 30:
                return "bending", neck_angle, threshold
            
            else:
                return "good", neck_angle, threshold
                
        except Exception as e:
            return "error", 0, self.adaptive_threshold

    def draw_posture_feedback(self, image, posture_type, angle, threshold):
        """Draw enhanced posture feedback on the image."""
        overlay = image.copy()
        height, width = image.shape[:2]
        
        # Draw semi-transparent header
        cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        # Different feedback based on posture type
        if posture_type == "good":
            cv2.putText(image, "EXCELLENT POSTURE!", (width//2 - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Neck Angle: {int(angle)} (Threshold: {int(threshold)})", 
                       (width//2 - 180, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Show streak and points
            cv2.putText(image, f"Streak: {self.streak_days} days | Points: {self.points}", 
                       (width//2 - 150, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2, cv2.LINE_AA)
        
        else:
            # Specific feedback for each posture issue
            if posture_type == "slouching":
                feedback_text = "SLOUCHING! SIT STRAIGHT!"
            elif posture_type == "bending":
                feedback_text = "BENDING TOO MUCH! SIT UPRIGHT!"
            elif posture_type == "lying":
                feedback_text = "LYING DOWN! SIT PROPERLY!"
            elif posture_type == "leaning":
                feedback_text = "LEANING TOO MUCH! CENTER YOURSELF!"
            else:
                feedback_text = "ADJUST YOUR POSTURE!"
            
            cv2.putText(image, feedback_text, (width//2 - 250, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Neck Angle: {int(angle)} (Need: {int(threshold)})", 
                       (width//2 - 180, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Show current tip for improvement
            if time.time() - self.last_tip_time > self.tip_interval:
                self.current_tip = (self.current_tip + 1) % len(self.correction_tips)
                self.last_tip_time = time.time()
            
            tip = self.correction_tips[self.current_tip]
            cv2.putText(image, f"Tip: {tip}", (width//2 - 300, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return image

    def update_posture_baseline(self, angle):
        """Update the user's personal posture baseline."""
        if self.user_posture_baseline is None:
            self.user_posture_baseline = angle
        else:
            # Slowly adapt to the user's typical good posture angle
            self.user_posture_baseline = 0.95 * self.user_posture_baseline + 0.05 * angle

    def run(self):
        """Main loop for the PostureGuard AI."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.pose.process(image)

            # Convert back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Get current time
            current_time = time.time()
            session_duration = current_time - time.mktime(self.session_start.timetuple())

            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Detect posture type
                    posture_type, angle, threshold = self.detect_posture(landmarks, image.shape)
                    
                    if posture_type != "error":
                        self.angle_history.append(angle)
                        smoothed_angle = sum(self.angle_history) / len(self.angle_history)
                        
                        # Record posture history for analytics
                        self.posture_data["posture_history"].append((session_duration, smoothed_angle))
                        
                        # Posture check
                        if posture_type != "good":
                            self.bad_posture_counter += 1
                            self.posture_data["bad_posture_time"] += 1
                            
                            if (self.bad_posture_counter > self.bad_posture_threshold and 
                                current_time - self.last_alert_time > self.alert_cooldown):
                                self.play_alert()  # This will play the beep sound
                                self.last_alert_time = current_time
                                self.bad_posture_counter = 0
                            
                            self.good_posture_time = 0
                            self.posture_data["posture_changes"] += 1
                        else:
                            # Update user's personal baseline when in good posture
                            self.update_posture_baseline(smoothed_angle)
                            
                            self.bad_posture_counter = 0
                            self.good_posture_time += 1
                            self.posture_data["good_posture_time"] += 1
                            
                            # Award points for maintaining good posture
                            if self.good_posture_time % 30 == 0:  # Every 30 frames of good posture
                                self.points += 1
                                if self.good_posture_time % 150 == 0:  # Every 150 frames
                                    self.play_success()
                        
                        # Draw feedback
                        image = self.draw_posture_feedback(image, posture_type, smoothed_angle, threshold)
                        
                        # Draw landmarks with custom styling
                        mp_drawing.draw_landmarks(
                            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
                        
                        # Display good posture time if applicable
                        if self.good_posture_time > 30:
                            minutes, seconds = divmod(self.good_posture_time // 30, 60)
                            cv2.putText(image, f"Good posture: {int(minutes)}m {int(seconds)}s", 
                                       (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            except Exception as e:
                cv2.putText(image, "Adjust position to be visible", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Add session info
            session_minutes = int(session_duration // 60)
            session_seconds = int(session_duration % 60)
            cv2.putText(image, f"Session: {session_minutes}m {session_seconds}s", 
                       (10, image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Add usage instructions
            cv2.putText(image, "Press 'q' to quit | 'p' for progress report | 'r' to reset", 
                       (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Display the frame
            cv2.imshow('Advanced PostureGuard AI - Unique Features', image)

            # Handle key presses
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                # Show progress chart
                chart = self.generate_progress_chart()
                if chart is not None:
                    cv2.imshow('Posture Progress Chart', chart)
                    cv2.waitKey(3000)  # Show for 3 seconds
                    cv2.destroyWindow('Posture Progress Chart')
            elif key == ord('r'):
                # Reset session
                self.posture_data = {
                    "good_posture_time": 0,
                    "bad_posture_time": 0,
                    "posture_changes": 0,
                    "posture_history": []
                }
                self.good_posture_time = 0
                print("Session reset!")

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        self.save_user_data()
        
        # Print session summary
        total_time = self.posture_data["good_posture_time"] + self.posture_data["bad_posture_time"]
        if total_time > 0:
            good_percentage = (self.posture_data["good_posture_time"] / total_time) * 100
            print(f"\nSession Summary:")
            print(f"Good posture: {good_percentage:.1f}% of the time")
            print(f"Posture changes: {self.posture_data['posture_changes']}")
            print(f"Points earned: {self.points}")
            print(f"Current streak: {self.streak_days} days")

if __name__ == "__main__":
    posture_guard = PostureGuardAI()
    posture_guard.run()