import cv2
from picamera2 import Picamera2
import time

def test_camera():
    print("Testing OpenCV with Picamera2...")
    
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
    
    # Save the frame
    cv2.imwrite('test_frame.jpg', frame)
    print("Saved test frame to 'test_frame.jpg'")
    
    # Display the frame
    cv2.imshow("Test Frame", frame)
    cv2.waitKey(2000)  # Show for 2 seconds
    
    # Clean up
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera test completed")

if __name__ == '__main__':
    test_camera() 