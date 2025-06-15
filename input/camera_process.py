import logging
from picamera2 import Picamera2

def camera_process(frame_queue, control_queue):
    """
    Camera process that captures frames and sends them to the main process.
    
    Args:
        frame_queue: Queue for sending frames to the main process
        control_queue: Queue for receiving control commands
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initializing camera...")
        picam2 = Picamera2()
        
        # Configure camera
        config = picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
        picam2.configure(config)
        picam2.start()
        logger.info("Camera started successfully")
        
        while True:
            # Check for stop command
            if not control_queue.empty():
                cmd = control_queue.get()
                if cmd == "STOP":
                    break
            
            # Capture and send frame
            frame = picam2.capture_array()
            frame_queue.put(frame)
            
    except Exception as e:
        logger.error(f"Camera process error: {e}", exc_info=True)
    finally:
        if 'picam2' in locals():
            picam2.stop()
        logger.info("Camera process stopped") 