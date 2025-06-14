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
    print("Camera started successfully")
    
    # Allow camera to warm up
    time.sleep(1)
    
    # Capture a frame
    frame = picam2.capture_array()
    
    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        print("Successfully detected hand landmarks")
        # Draw the hand landmarks
        mp_draw = mp.solutions.drawing_utils
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
    else:
        print("No hand detected in frame")
    
    # Save the frame
    cv2.imwrite('test_mediapipe.jpg', frame)
    print("Saved test frame to 'test_mediapipe.jpg'")
    
    # Display the frame
    cv2.imshow("MediaPipe Test", frame)
    cv2.waitKey(2000)  # Show for 2 seconds
    
    # Clean up
    picam2.stop()
    cv2.destroyAllWindows()
    print("MediaPipe test completed")

if __name__ == '__main__':
    test_mediapipe() 