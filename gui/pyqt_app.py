from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
import cv2
from PIL import Image
from config.config import load_config
from questions import QUESTIONS
from input.asl_detector import ASLDetector
from input.speech_recognizer import SpeechRecognizer
from nlp.answer_normalizer import normalize
from nlp.text_processor import TextProcessor
from kb.tax_engine import TaxEngine
from tts.tts_engine import TTSEngine
from collections import deque
import speech_recognition as sr
import sys
import os
import time
import logging
import numpy as np
import mediapipe as mp
from queue import Queue
from input.camera_process import camera_process
from input.llm_processor import LLMProcessor

class SpeechRecognitionThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, recognizer):
        super().__init__()
        self.recognizer = recognizer
        
    def run(self):
        try:
            raw = self.recognizer.listen()
            if raw:
                self.finished.emit(raw)
            else:
                self.error.emit("No speech detected")
        except sr.WaitTimeoutError:
            self.error.emit("No speech detected")
        except Exception as e:
            self.error.emit(str(e))

class MicrophoneIndicator(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(100, 100)
        self.setMaximumSize(100, 100)
        self.alpha = 0
        self.fade_in = True
        self.setStyleSheet("background-color: transparent;")
        self.is_listening = False
        self.active_color = QColor(0, 255, 0)  # Green for active
        self.inactive_color = QColor(150, 150, 150) # Gray for inactive
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Determine color based on listening state
        current_color = self.active_color
        current_alpha = self.alpha
        if not self.is_listening:
            current_color = self.inactive_color
            current_alpha = 255 # Solid gray when not listening
            
        # Draw microphone icon
        center_x = self.width() // 2
        center_y = self.height() // 2
        
        # Set color with alpha for fade effect
        color = QColor(current_color.red(), current_color.green(), current_color.blue(), current_alpha)
        painter.setPen(color)
        painter.setBrush(color)
        
        # Draw microphone body
        painter.drawRect(center_x - 15, center_y - 25, 30, 40)
        
        # Draw microphone stand
        painter.drawRect(center_x - 5, center_y + 15, 10, 15)
        
        # Draw microphone base
        painter.drawEllipse(center_x - 20, center_y + 30, 40, 10)
        
        # Draw sound waves only if listening
        if self.is_listening:
            wave_color = QColor(current_color.red(), current_color.green(), current_color.blue(), current_alpha // 2)
            painter.setPen(wave_color)
            painter.setBrush(Qt.NoBrush)
            
            # Draw three arcs
            for i in range(3):
                radius = 30 + (i * 15)
                painter.drawArc(center_x - radius, center_y - radius, 
                              radius * 2, radius * 2, 0, 5760)  # 5760 = 360 * 16

    def update_animation(self):
        if not self.is_listening:
            self.alpha = 0 # Reset alpha when not listening
            self.fade_in = True
            self.update()
            return
            
        if self.fade_in:
            self.alpha += 5
            if self.alpha >= 255:
                self.alpha = 255
                self.fade_in = False
        else:
            self.alpha -= 5
            if self.alpha <= 0:
                self.alpha = 0
                self.fade_in = True
        self.update()

    def set_listening(self, listening):
        self.is_listening = listening
        if not listening:
            self.alpha = 255 # Make it solid gray when not listening
            self.fade_in = True # Reset fade for next time it becomes active
        else:
            self.alpha = 0 # Start from transparent for fade-in
            self.fade_in = True
        self.update()

class SmartTaxAdvisor(QMainWindow):
    def __init__(self, frame_queue=None, control_queue=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Initialize queues with defaults if not provided
        self.frame_queue = frame_queue if frame_queue is not None else Queue()
        self.control_queue = control_queue if control_queue is not None else Queue()
        
        # Initialize components
        self.asl = ASLDetector()
        self.speech = SpeechRecognizer()
        self.llm = LLMProcessor()
        
        # Initialize state variables
        self.current_q = 0
        self.answers = {}
        self.current_mode = None
        self.showing_answer = False
        self.answer_timer = QTimer()
        self.answer_timer.timeout.connect(self.hide_answer)
        
        # ASL detection state
        self.confidence_threshold = 0.6
        self.confidence_duration = 1.0
        self.confidence_start_time = None
        self.last_detected_label = None
        self.current_confidence = 0.0
        
        # Digit input state
        self.asking_digits = False
        self.asking_digit_count = False
        self.current_number = ""
        self.expected_digits = 0
        self.can_accept_digit = True
        self.no_hand_frames = 0
        
        # Set up UI
        self.setup_ui()
        
        # Set up frame processing timer
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.process_frame)
        self.frame_timer.start(30)  # 30ms = ~33 FPS
        
        # Set window properties
        self.setWindowTitle("Smart Tax Advisor")
        self.showFullScreen()
        
    def process_frame(self):
        """Process incoming frames from the camera"""
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                if frame is not None:
                    if self.current_mode == "sign":
                        processed_frame = self.process_sign_frame(frame)
                    else:
                        processed_frame = frame
                    
                    # Convert frame to QImage and display
                    height, width, channel = processed_frame.shape
                    bytes_per_line = 3 * width
                    q_image = QImage(processed_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    self.video_label.setPixmap(QPixmap.fromImage(q_image))
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
    
    def process_sign_frame(self, frame):
        """Process frame for sign language detection"""
        if self.showing_answer:
            return frame
            
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Determine which model to use
            current_question = QUESTIONS[self.current_q - 1] if self.current_q > 0 else None
            use_digit_model = (
                current_question and 
                (current_question.get('type') == 'digits' or 
                 'children' in current_question.get('text', '').lower())
            )
            
            # Process with appropriate detector
            if use_digit_model:
                label, conf = self.safe_asl_recognition(rgb_frame, use_digit_model=True)
                
                # Simplified hand tracking logic
                if label is None:
                    self.no_hand_frames += 1
                    if self.no_hand_frames >= 5:  # Reduced threshold
                        self.can_accept_digit = True
                else:
                    self.no_hand_frames = 0
                    # Only process if we can accept digits and confidence is high
                    if self.can_accept_digit and conf >= self.confidence_threshold:
                        self.handle_digit_detection(label, conf)
                        self.can_accept_digit = False  # Prevent multiple detections
            else:
                label, conf = self.safe_asl_recognition(rgb_frame, use_digit_model=False)
                
                # Consistent confidence handling for letters
                if conf >= self.confidence_threshold:
                    if self.confidence_start_time is None:
                        self.confidence_start_time = time.time()
                        self.last_detected_label = label
                    elif label == self.last_detected_label:
                        elapsed_time = time.time() - self.confidence_start_time
                        if elapsed_time >= self.confidence_duration:
                            self.handle_letter_detection(label, conf)
                            self.confidence_start_time = None
                            self.last_detected_label = None
                    else:
                        # Different label detected, reset timer
                        self.confidence_start_time = time.time()
                        self.last_detected_label = label
                else:
                    self.confidence_start_time = None
                    self.last_detected_label = None
            
            # Update display
            self.update_confidence_display(conf)
            return frame
            
        except Exception as e:
            self.logger.error(f"Error processing sign frame: {e}")
            return frame
    
    def handle_digit_detection(self, label, confidence):
        """Handle detected digit input"""
        if not label.isdigit():
            return
            
        if self.asking_digit_count:
            # Handle digit count input
            digit_count = int(label)
            if 1 <= digit_count <= 9:  # Reasonable range
                self.expected_digits = digit_count
                self.asking_digit_count = False
                self.current_number = ""
                self.feedback_label.setText(f"Please enter {self.expected_digits} digits")
                self.number_feedback.setText("")
                self.number_feedback.show()
            else:
                self.feedback_label.setText("Please enter a number between 1 and 9")
        else:
            # Handle actual number input
            if len(self.current_number) < self.expected_digits:
                self.current_number += label
                self.feedback_label.setText(f"Detected digit: {label}")
                self.number_feedback.setText(self.current_number)
                
                if len(self.current_number) == self.expected_digits:
                    self.answers[self.current_q] = self.current_number
                    self.asking_digits = False
                    self.show_answer(self.current_number)
    
    def handle_letter_detection(self, label, confidence):
        """Handle detected letter input"""
        if label.upper() in ['Y', 'N']:
            self.answers[self.current_q] = label.upper()
            self.show_answer(label.upper())
            self.feedback_label.setText(f"Detected: {label.upper()}")
    
    def update_confidence_display(self, confidence):
        """Update confidence display consistently"""
        self.current_confidence = confidence
        if confidence > 0:
            self.confidence_label.setText(f"Confidence: {confidence:.2%}")
            # Color code confidence
            if confidence >= 0.8:
                color = "#4CAF50"  # Green
            elif confidence >= 0.6:
                color = "#FF9800"  # Orange
            else:
                color = "#F44336"  # Red
            
            self.confidence_label.setStyleSheet(f"""
                font-size: 14px;
                color: {color};
                padding: 5px;
                background-color: #f0f0f0;
                border-radius: 5px;
                font-weight: bold;
            """)
        else:
            self.confidence_label.setText("No detection")
            self.confidence_label.setStyleSheet("""
                font-size: 14px;
                color: #666;
                padding: 5px;
                background-color: #f0f0f0;
                border-radius: 5px;
            """)
    
    def safe_asl_recognition(self, frame, use_digit_model=False):
        """Safely perform ASL recognition with error handling"""
        try:
            if use_digit_model:
                if hasattr(self.asl, 'recognize_digit'):
                    return self.asl.recognize_digit(frame)
                else:
                    self.logger.warning("recognize_digit method not found")
                    return None, 0.0
            else:
                if hasattr(self.asl, 'recognize_letter'):
                    return self.asl.recognize_letter(frame)
                else:
                    self.logger.warning("recognize_letter method not found")
                    return None, 0.0
        except Exception as e:
            self.logger.error(f"ASL recognition error: {e}")
            return None, 0.0
    
    def ask_next(self):
        """Ask the next question with proper setup"""
        if self.current_q < len(QUESTIONS):
            question = QUESTIONS[self.current_q]
            self.question_label.setText(question['text'])
            
            # Clear previous state
            self.confidence_start_time = None
            self.last_detected_label = None
            self.feedback_label.setText("")
            
            # Set up for digit input
            if question.get('type') == 'digits' or 'children' in question.get('text', '').lower():
                self.setup_digit_input()
            else:
                self.setup_letter_input()
            
            self.current_q += 1
        else:
            self.finish()
    
    def setup_digit_input(self):
        """Set up for digit input questions"""
        self.asking_digits = True
        self.asking_digit_count = True
        self.current_number = ""
        self.expected_digits = 0
        self.can_accept_digit = True
        self.no_hand_frames = 0
        
        self.feedback_label.setText("Show the number of digits you'll enter")
        self.number_feedback.setText("")
        self.number_feedback.show()
    
    def setup_letter_input(self):
        """Set up for letter input questions"""
        self.asking_digits = False
        self.asking_digit_count = False
        self.number_feedback.hide()
        self.feedback_label.setText("Show Y for Yes or N for No")
    
    def show_answer(self, answer):
        """Show the answer for 2 seconds before moving to next question"""
        self.showing_answer = True
        self.answer_label.setText(f"Answer: {answer}")
        self.answer_label.show()
        self.feedback_label.setText("")
        self.confidence_label.setText("")
        self.number_feedback.hide()
        self.answer_timer.start(2000)  # 2 seconds
    
    def hide_answer(self):
        """Hide the answer after 2 seconds"""
        self.showing_answer = False
        self.answer_label.hide()
        self.ask_next()
    
    def finish(self):
        """Handle completion of questions"""
        self.video_label.hide()
        
        # Process answers and display results
        result = self.engine.process_answers(self.answers)
        self.result_text.setText(result)
        self.results_widget.show()

def main():
    app = QApplication(sys.argv)
    window = SmartTaxAdvisor()
    window.show()
    sys.exit(app.exec()) 