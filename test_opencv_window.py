import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("Attempting to create a blank image...")
# Create a black image
img = np.zeros((480, 640, 3), dtype=np.uint8)
logger.debug("Image created.")

try:
    logger.debug("Attempting to create a window...")
    cv2.namedWindow("Simple Test", cv2.WINDOW_NORMAL)
    logger.debug("Window created successfully.")

    logger.debug("Attempting to show image...")
    cv2.imshow("Simple Test", img)
    logger.debug("Image shown successfully.")

    logger.debug("Waiting for a key press for 5 seconds...")
    cv2.waitKey(5000)  # Wait for 5 seconds
    logger.debug("Wait finished.")

except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True)

finally:
    logger.debug("Destroying all windows.")
    cv2.destroyAllWindows()
    logger.debug("Script finished.")