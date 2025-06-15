# app.py
import sys
import logging
from multiprocessing import Process, Queue
from PyQt5.QtWidgets import QApplication
from gui.pyqt_app import SmartTaxAdvisor

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create queues for inter-process communication
    frame_queue = Queue(maxsize=2)
    control_queue = Queue()
    
    # Start the camera process
    from input.camera_process import camera_process
    p_camera = Process(target=camera_process, args=(frame_queue, control_queue))
    p_camera.start()
    
    # Start the GUI application
    app = QApplication(sys.argv)
    window = SmartTaxAdvisor(frame_queue, control_queue)
    window.show()
    
    # Run the application
    exit_code = app.exec_()
    
    # Clean up
    logger.info("Application finished. Terminating camera process...")
    control_queue.put("STOP")
    p_camera.join()
    
    sys.exit(exit_code)

if __name__ == '__main__':
    main()