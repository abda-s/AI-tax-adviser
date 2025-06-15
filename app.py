# app.py
import multiprocessing as mp
from gui.pyqt_app import SmartTaxAdvisor
from input.camera_process import camera_process
import sys
import os
import signal
import logging
from PyQt5.QtWidgets import QApplication

# Set Qt platform for Raspberry Pi
os.environ["QT_QPA_PLATFORM"] = "eglfs"

def signal_handler(signum, frame):
    """Handle termination signals"""
    print("\nReceived termination signal. Cleaning up...")
    sys.exit(0)

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create queues for inter-process communication
        frame_queue = mp.Queue(maxsize=10)
        control_queue = mp.Queue()
        
        # Start camera process
        camera_proc = mp.Process(target=camera_process, args=(frame_queue, control_queue))
        camera_proc.start()
        logger.info("Camera process started")
        
        # Create Qt application
        app = QApplication(sys.argv)
        
        # Create main window
        window = SmartTaxAdvisor(frame_queue, control_queue)
        window.show()
        
        # Run application
        exit_code = app.exec_()
        
        # Cleanup
        logger.info("Application shutting down...")
        control_queue.put("STOP")
        camera_proc.join(timeout=5)
        if camera_proc.is_alive():
            camera_proc.terminate()
        
        return exit_code
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())