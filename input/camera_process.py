import multiprocessing as mp
from picamera2 import Picamera2
import time
import logging
import os

def camera_process(frame_queue, control_queue):
    """
    Camera process that captures frames and sends them to the main process.
    
    Args:
        frame_queue: Queue for sending frames to the main process
        control_queue: Queue for receiving control commands
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Initializing camera...")
    
    # Try to initialize camera with retries
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Configure camera
            picam2 = Picamera2()
            
            # Set camera configuration
            config = picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"},
                lores={"size": (320, 240), "format": "YUV420"}
            )
            picam2.configure(config)
            
            # Start camera
            picam2.start()
            logger.info("Camera initialized successfully")
            break
            
        except Exception as e:
            logger.error(f"Camera initialization attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Failed to initialize camera after all attempts")
                return
    
    try:
        while True:
            # Check for control messages
            if not control_queue.empty():
                msg = control_queue.get()
                if msg == "STOP":
                    break
            
            # Capture frame
            frame = picam2.capture_array()
            
            # Put frame in queue
            if not frame_queue.full():
                frame_queue.put(frame)
            
            # Small delay to prevent overwhelming the queue
            time.sleep(0.01)
            
    except Exception as e:
        logger.error(f"Camera process error: {e}")
    finally:
        try:
            picam2.stop()
            logger.info("Camera stopped")
        except:
            pass
        logger.info("Camera process stopped") 