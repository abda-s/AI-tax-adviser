import cv2
import time
import logging
from multiprocessing import Process, Queue

# --- Process 1: Camera Capture ---
# This process's only job is to capture ONE frame and send it.
def camera_process(queue):
    # These imports happen only inside this process
    from picamera2 import Picamera2

    try:
        print("[Camera Process] Initializing camera...")
        picam2 = Picamera2()
        
        # We use the "RGB888" format that we know gives a correct BGR image on your system.
        # This is what cv2.imwrite and cv2.imshow expect.
        config = picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
        
        picam2.configure(config)
        picam2.start()
        print("[Camera Process] Camera started, warming up...")
        time.sleep(1) # Allow camera to warm up

        # Capture a single frame
        frame = picam2.capture_array()
        print("[Camera Process] Frame captured.")
        
        # Put the single frame into the queue for the other process
        queue.put(frame)
        print("[Camera Process] Frame sent to display process.")

    except Exception as e:
        print(f"[Camera Process] Error: {e}")
    finally:
        # Clean up the camera immediately after its job is done
        if 'picam2' in locals():
            picam2.stop()
        print("[Camera Process] Camera stopped. Process finished.")


# --- Process 2: Save and Display ---
# This process receives the frame, saves it, and shows it.
def display_process(queue):
    try:
        print("[Display Process] Waiting for frame from camera process...")
        # Get the single frame from the queue
        frame = queue.get()
        print("[Display Process] Frame received.")

        # Save the frame to a file
        cv2.imwrite('test_frame.jpg', frame)
        print("[Display Process] Saved test frame to 'test_frame.jpg'")

        # Display the frame in a window
        cv2.imshow("Test Frame", frame)
        print("[Display Process] Displaying frame for 2 seconds...")
        cv2.waitKey(2000)

    except Exception as e:
        print(f"[Display Process] Error: {e}")
    finally:
        # Clean up the OpenCV window
        cv2.destroyAllWindows()
        print("[Display Process] Window closed. Process finished.")


# --- Main block to start and manage the processes ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting single-frame capture test...")
    frame_queue = Queue(maxsize=1)

    # Create the two processes
    p_camera = Process(target=camera_process, args=(frame_queue,))
    p_display = Process(target=display_process, args=(frame_queue,))

    # Start the processes
    p_camera.start()
    p_display.start()

    # Wait for both processes to complete their tasks
    p_camera.join()
    p_display.join()

    logger.info("Application finished successfully, terminal should not freeze.")