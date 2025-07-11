import cv2
import mediapipe as mp
import time
import logging
from multiprocessing import Process, Queue

# --- Process 1: Camera Capture (No changes needed here) ---
def camera_process(queue):
    from picamera2 import Picamera2
    try:
        print("[Camera Process] Initializing camera...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
        picam2.configure(config)
        picam2.start()
        print("[Camera Process] Camera started.")
        while True:
            frame = picam2.capture_array()
            queue.put(frame)
    except Exception as e:
        print(f"[Camera Process] Error: {e}")
    finally:
        if 'picam2' in locals():
            picam2.stop()
        print("[Camera Process] Stopped.")


# --- Process 2: MediaPipe and Display ---
def display_process(queue):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    try:
        print("[Display Process] Creating OpenCV window...")
        cv2.namedWindow("MediaPipe Hand Detection", cv2.WINDOW_NORMAL)
        print("[Display Process] Window created.")

        while True:
            # Get a read-only BGR frame from the queue
            frame_bgr = queue.get()
            if frame_bgr is None:
                break

            ### THE FIX IS HERE: Create a writeable copy of the frame ###
            frame_bgr = frame_bgr.copy()

            # Now we can safely proceed with the rest of the logic
            frame_bgr.flags.writeable = False
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            frame_bgr.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame_bgr,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
            
            cv2.imshow("MediaPipe Hand Detection", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"[Display Process] Error: {e}")
    finally:
        hands.close()
        cv2.destroyAllWindows()
        print("[Display Process] Window closed.")


# --- Main block (No changes needed here) ---
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