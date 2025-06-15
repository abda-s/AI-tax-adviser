import cv2
import time
import logging
from multiprocessing import Process, Queue

# --- Process 1: Camera Capture ---
# This process gets the frame from the camera.
def camera_process(queue):
    from picamera2 import Picamera2

    try:
        print("[Camera Process] Initializing camera...")
        picam2 = Picamera2()
        
        # Request the format that you confirmed gives correct colors directly.
        # On your system, "RGB888" outputs BGR.
        config = picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
        
        picam2.configure(config)
        picam2.start()
        print("[Camera Process] Camera started.")

        while True:
            # Capture the frame and send it to the display process.
            frame = picam2.capture_array()
            queue.put(frame)

    except Exception as e:
        print(f"[Camera Process] Error: {e}")
    finally:
        if 'picam2' in locals():
            picam2.stop()
        print("[Camera Process] Stopped.")


# --- Process 2: Display Window ---
# This process displays the frame exactly as it receives it.
def display_process(queue):
    try:
        print("[Display Process] Creating OpenCV window...")
        cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
        print("[Display Process] Window created.")

        while True:
            # Get a frame from the queue.
            frame = queue.get()

            if frame is None:
                break

            # ### THE FIX ###
            # Display the frame directly, with NO color conversion.
            cv2.imshow("Camera Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"[Display Process] Error: {e}")
    finally:
        cv2.destroyAllWindows()
        print("[Display Process] Window closed.")


# --- Main block to start and manage the processes ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting multi-process application...")
    frame_queue = Queue(maxsize=2)

    p_camera = Process(target=camera_process, args=(frame_queue,))
    p_display = Process(target=display_process, args=(frame_queue,))

    p_camera.start()
    p_display.start()

    p_display.join()

    logger.info("Display process finished. Terminating camera process...")
    p_camera.terminate()
    p_camera.join()

    logger.info("Application finished.")