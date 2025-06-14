import cv2
import mediapipe as mp
from picamera2 import Picamera2
import time

def test_mediapipe():
    print("Testing MediaPipe hand detection with Picamera2...")
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize Picamera2
    picam2 = Picamera2()
    
    # Configure the camera
    config = picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": (640, 480)}
    )
    picam2.configure(config)
    
    # Start the camera
    picam2.start()
    print("Camera started successfully!")
    print("A window will open with the live feed.")
    print("Press 'q' in the camera window to quit.")
    
    # Allow camera to warm up
    time.sleep(1)
    
    # Initialize drawing utilities
    mp_draw = mp.solutions.drawing_utils
    
    # Loop to continuously get frames from the camera
    while True:
        # Capture a frame
        frame = picam2.capture_array()
        
        # Convert the frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
                # Add text to show hand is detected
                cv2.putText(frame, "Hand Detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow("MediaPipe Hand Detection - Press 'q' to Quit", frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # If 'q' is pressed, break the loop
        if key == ord("q"):
            break
    
    # Clean up
    print("Closing camera and cleaning up...")
    picam2.stop()
    cv2.destroyAllWindows()
    print("Test completed.")

if __name__ == '__main__':
    test_mediapipe() 