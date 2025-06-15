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
import numpy as np
import mediapipe as mp
import logging
import os
import time

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
    def __init__(self, frame_queue, control_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.control_queue = control_queue
        self.logger = logging.getLogger(__name__)
        
        # Set window flags for Raspberry Pi
        if os.environ.get("QT_QPA_PLATFORM") == "eglfs":
            self.setWindowFlags(Qt.FramelessWindowHint)
            # Set fixed size for Raspberry Pi display
            self.setFixedSize(800, 600)
            # Center the window on screen
            screen = QApplication.primaryScreen().geometry()
            x = (screen.width() - self.width()) // 2
            y = (screen.height() - self.height()) // 2
            self.move(x, y)
        else:
            self.setGeometry(100, 100, 800, 600)
        
        self.setWindowTitle('Smart Tax Advisor')
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)
        self.layout.setAlignment(Qt.AlignCenter)
        
        # Load modules
        cfg = load_config()
        self.asl = ASLDetector(
            cfg['asl_model'], cfg['asl_map'],
            cfg['digit_model'], cfg['digit_map'],
            cfg['confidence_threshold']
        )
        self.sr = SpeechRecognizer()
        self.tts = TTSEngine()
        self.engine = TaxEngine()
        self.text_processor = TextProcessor()
        
        # Initialize state variables
        self.current_q = 0
        self.answers = {}
        self.selected_mode = None
        self.current_number = ""
        self.expected_digits = 0
        self.no_hand_frames = 0
        self.can_accept_digit = True
        self.asking_digits = False
        self.is_listening = False
        self.is_capturing = False
        self.asking_digit_count = False  # New state for asking number of digits
        
        # Confidence tracking
        self.current_confidence = 0.0
        self.confidence_start_time = None
        self.confidence_threshold = 0.75
        self.confidence_duration = 0.3  # seconds
        self.last_detected_label = None
        
        # Hand tracking
        self.hand_removed = False
        self.hand_removed_frames = 0
        self.hand_removed_threshold = 10  # frames
        
        # Answer display timer
        self.answer_timer = QTimer()
        self.answer_timer.setSingleShot(True)
        self.answer_timer.timeout.connect(self.show_next_question)
        self.showing_answer = False
        
        # Create widgets
        self.setup_ui()
        
        # Set up frame processing timer
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.process_frame)
        self.frame_timer.start(30)  # Process frames at ~30 FPS
        
        # Create timers
        self.listening_timer = QTimer(self)
        self.listening_timer.timeout.connect(self.update_listening_animation)
        self.mic_animation_timer = QTimer(self)
        self.mic_animation_timer.timeout.connect(self.mic_indicator.update_animation)
        self.mic_animation_timer.start(50)  # Update every 50ms for smooth animation
        
        # Set initial status for speech mode
        self.update_status("Not Listening", "gray")
        self.mic_indicator.set_listening(False)

        # Add escape key handler for fullscreen mode
        if os.environ.get("QT_QPA_PLATFORM") == "eglfs":
            QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == event.KeyPress and event.key() == Qt.Key_Escape:
            self.close()
            return True
        return super().eventFilter(obj, event)

    def setup_ui(self):
        # Create container widgets for different modes
        self.mode_selection_widget = QWidget()
        self.mode_selection_layout = QVBoxLayout(self.mode_selection_widget)
        
        self.asl_widget = QWidget()
        self.asl_layout = QVBoxLayout(self.asl_widget)
        
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        
        # Mode selection
        self.mode_label = QLabel("Select Input Mode:")
        self.mode_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        
        # Buttons
        self.sign_btn = QPushButton("Sign Language")
        self.sign_btn.clicked.connect(self.select_sign_mode)
        self.sign_btn.setFixedSize(250, 150)
        self.sign_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 20px;
                font-size: 20px;
                padding: 10px 20px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        self.speech_btn = QPushButton("Speech Recognition")
        self.speech_btn.clicked.connect(self.select_speech_mode)
        self.speech_btn.setFixedSize(250, 150)
        self.speech_btn.setStyleSheet("""
            QPushButton {
                background-color: #008CBA;
                color: white;
                border-radius: 20px;
                font-size: 20px;
                padding: 10px 20px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #007bb5;
            }
        """)
        
        # Layout for buttons
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignCenter)
        button_layout.addWidget(self.sign_btn)
        button_layout.addWidget(self.speech_btn)
        
        # Add widgets to mode selection layout
        self.mode_selection_layout.addWidget(self.mode_label)
        self.mode_selection_layout.addLayout(button_layout)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        
        # Question and status labels
        self.question_label = QLabel()
        self.question_label.setStyleSheet("font-size: 16px;")
        
        # Status label
        self.status_label = QLabel()
        self.status_label.setStyleSheet("font-size: 14px; color: #666;")
        
        # Confidence display
        self.confidence_label = QLabel()
        self.confidence_label.setStyleSheet("""
            font-size: 14px;
            color: #666;
            padding: 5px;
            background-color: #f0f0f0;
            border-radius: 5px;
        """)
        
        # Feedback label
        self.feedback_label = QLabel()
        self.feedback_label.setStyleSheet("""
            font-size: 16px;
            color: #4CAF50;
            padding: 5px;
            background-color: #e8f5e9;
            border-radius: 5px;
        """)
        
        # Number input feedback
        self.number_feedback = QLabel()
        self.number_feedback.setStyleSheet("""
            font-size: 48px;
            color: #2196F3;
            padding: 20px;
            background-color: #E3F2FD;
            border-radius: 15px;
            margin: 10px;
            font-weight: bold;
        """)
        self.number_feedback.setAlignment(Qt.AlignCenter)
        self.number_feedback.setMinimumHeight(100)
        self.number_feedback.hide()
        
        # Answer display label
        self.answer_label = QLabel()
        self.answer_label.setStyleSheet("""
            font-size: 24px;
            color: #2196F3;
            padding: 10px;
            background-color: #E3F2FD;
            border-radius: 10px;
            margin: 10px;
        """)
        self.answer_label.setAlignment(Qt.AlignCenter)
        self.answer_label.hide()
        
        # Result text
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("""
            font-size: 14px;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 5px;
        """)
        
        # Create microphone indicator
        self.mic_indicator = MicrophoneIndicator()
        self.mic_indicator.hide()
        
        # Add widgets to ASL layout
        self.asl_layout.addWidget(self.video_label)
        self.asl_layout.addWidget(self.question_label)
        self.asl_layout.addWidget(self.status_label)
        self.asl_layout.addWidget(self.confidence_label)
        self.asl_layout.addWidget(self.feedback_label)
        self.asl_layout.addWidget(self.number_feedback)
        self.asl_layout.addWidget(self.answer_label)
        
        # Add result text to results layout
        self.results_layout.addWidget(self.result_text)
        
        # Add all widgets to main layout
        self.layout.addWidget(self.mode_selection_widget)
        self.layout.addWidget(self.asl_widget)
        self.layout.addWidget(self.results_widget)
        self.layout.addWidget(self.mic_indicator, alignment=Qt.AlignCenter)
        
        # Initially hide ASL widget and results widget
        self.asl_widget.hide()
        self.results_widget.hide()

    def process_frame(self):
        """Process frames from the camera queue"""
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            
            if self.selected_mode == "sign":
                # Process frame for sign language
                frame = self.process_sign_frame(frame)
            elif self.selected_mode == "speech":
                # Process frame for speech mode (just display)
                pass
            
            # Convert frame to QImage and display
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_image))
    
    def process_sign_frame(self, frame):
        """Process frame for sign language detection"""
        if frame is None:
            return
            
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.mp_hands.process(rgb_frame)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Get hand position
            hand_landmarks = results.multi_hand_landmarks[0]
            x = hand_landmarks.landmark[8].x
            y = hand_landmarks.landmark[8].y
            
            # Update hand tracking
            if self.asking_digit_count:
                self.update_hand_tracking(x, y)
            
            # Process sign detection
            if self.asking_digit_count:
                self.process_digit_count_input(results)
            else:
                self.process_sign_input(results)
        else:
            # No hand detected
            self.no_hand_frames += 1
            if self.no_hand_frames >= self.hand_removed_threshold:
                self.hand_removed = True
                self.can_accept_digit = True
                self.confidence_label.setText("")
                self.feedback_label.setText("")
        
        # Convert frame back to RGB for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))
        
    def update_hand_tracking(self, x, y):
        """Update hand tracking state"""
        if not self.hand_removed:
            self.no_hand_frames = 0
            return
            
        # Check if hand is in the center region
        if 0.3 <= x <= 0.7 and 0.3 <= y <= 0.7:
            self.hand_removed = False
            self.no_hand_frames = 0
            self.can_accept_digit = False
            
    def process_digit_count_input(self, results):
        """Process input for digit count"""
        if not self.hand_removed or not self.can_accept_digit:
            return
            
        # Get prediction from ASL detector
        label, confidence = self.asl.detect_digit(results)
        
        if label is not None and confidence > self.confidence_threshold:
            try:
                digit = int(label)
                if 1 <= digit <= 9:  # Only accept digits 1-9
                    self.expected_digits = digit
                    self.asking_digit_count = False
                    self.current_number = []
                    self.hand_removed = True
                    self.can_accept_digit = True
                    self.feedback_label.setText(f"Please enter {digit} digits")
                    self.number_feedback.setText("")
                    self.number_feedback.show()
            except ValueError:
                pass
                
    def process_sign_input(self, results):
        """Process sign input for letters or digits"""
        if not self.hand_removed or not self.can_accept_digit:
            return
            
        # Get prediction from appropriate detector
        if self.using_digit_model:
            label, confidence = self.asl.detect_digit(results)
        else:
            label, confidence = self.asl.detect_letter(results)
            
        if label is not None and confidence > self.confidence_threshold:
            self.handle_sign_detection(label, confidence)
            
    def handle_sign_detection(self, label, confidence):
        """Handle detected sign"""
        if self.using_digit_model:
            # Handle digit input
            try:
                digit = int(label)
                if 0 <= digit <= 9:  # Accept digits 0-9
                    self.current_number.append(str(digit))
                    current = ''.join(self.current_number)
                    self.feedback_label.setText(f"Detected digit: {digit}")
                    self.number_feedback.setText(current)
                    
                    if len(self.current_number) >= self.expected_digits:
                        self.handle_complete_number()
            except ValueError:
                pass
        else:
            # Handle letter input
            self.current_answer += label
            self.feedback_label.setText(f"Detected: {label}")
            self.confidence_label.setText(f"Confidence: {confidence:.2f}")
            
    def handle_complete_number(self):
        """Handle completed number input"""
        number = ''.join(self.current_number)
        self.answer_label.setText(f"Number: {number}")
        self.answer_label.show()
        self.number_feedback.hide()
        
        # Process the answer
        self.process_answer(number)
        
        # Reset state
        self.reset_digit_state()
        
    def reset_digit_state(self):
        """Reset all digit-related state variables"""
        self.asking_digit_count = True
        self.expected_digits = 0
        self.current_number = []
        self.hand_removed = True
        self.can_accept_digit = True
        self.no_hand_frames = 0
        self.feedback_label.setText("")
        self.confidence_label.setText("")
        self.answer_label.hide()
        
    def ask_next(self):
        """Ask the next question"""
        if self.current_q < len(QUESTIONS):
            question = QUESTIONS[self.current_q]
            self.question_label.setText(question['text'])
            
            # Reset state for new question
            self.reset_digit_state()
            
            # Determine if we should use digit model
            self.using_digit_model = (
                "digit" in question['text'].lower() or 
                "children" in question['text'].lower() or
                "number" in question['text'].lower()
            )
            
            if self.using_digit_model:
                self.feedback_label.setText("Please show the number of digits (1-9)")
                self.number_feedback.show()
            else:
                self.current_answer = ""
                self.feedback_label.setText("")
                self.number_feedback.hide()
                
            self.current_q += 1
        else:
            self.finish()
    
    def finish(self):
        """Handle completion of questions"""
        self.asl_widget.hide()
        
        # Process answers and display results
        result = self.engine.process_answers(self.answers)
        self.result_text.setText(result)
        self.results_widget.show()

    def update_listening_animation(self):
        """Update the listening animation dots"""
        if self.selected_mode == 'speech' and self.is_listening:
            self.listening_dots = (self.listening_dots + 1) % 4
            dots = "." * self.listening_dots
            self.status_label.setText(f"Listening{dots}")
            self.status_label.setStyleSheet("font-size: 12px; color: green; font-weight: bold;")
        elif self.selected_mode == 'speech':
            self.status_label.setText("Not Listening")
            self.status_label.setStyleSheet("font-size: 12px; color: gray;")

    def update_status(self, message, color='blue'):
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"font-size: 12px; color: {color};")

    def process_speech_result(self, raw):
        """Process the speech recognition result"""
        q = QUESTIONS[self.current_q]
        self.update_status(f"Raw text: {raw}", "blue")
        self.result_text.setText(f"Raw text: {raw}\n")
        
        processed_text = self.text_processor.process_text(raw, q['type'])
        self.result_text.append(f"Processed text: {processed_text}\n")
        
        if q['type'] == 'number' and 'salary' in q['text'].lower():
            if processed_text.isdigit() and 1 <= int(processed_text) <= 99999:
                self.answers[q['id']] = processed_text
                self.current_q += 1
                self.is_listening = False
                self.update_status("")
                QTimer.singleShot(1500, self.ask_next)
            else:
                self.update_status("Please say a valid number between 1 and 99999", "red")
                self.result_text.append("Please say a valid number between 1 and 99999")
                self.is_listening = False
                QTimer.singleShot(1000, self.capture_speech_answer)
        else:
            self.answers[q['id']] = processed_text
            self.current_q += 1
            self.is_listening = False
            self.update_status("")
            QTimer.singleShot(1500, self.ask_next)
        
        self.mic_indicator.set_listening(False) # Stop microphone animation
        self.update_status("Not Listening", "gray") # Update status

    def handle_speech_error(self, error_msg):
        """Handle speech recognition errors"""
        self.listening_timer.stop()
        self.mic_indicator.set_listening(False)
        self.update_status("Not Listening", "gray") # Update status
        self.result_text.setText(f"Error: {error_msg}")
        self.is_listening = False
        QTimer.singleShot(1000, self.capture_speech_answer)

    def capture_speech_answer(self):
        if self.is_listening:
            return

        self.is_listening = True
        self.listening_dots = 0
        self.listening_timer.start(500)  # Update animation every 500ms
        self.mic_indicator.set_listening(True)  # Start microphone animation
        
        # Create and start speech recognition thread
        self.speech_thread = SpeechRecognitionThread(self.sr)
        self.speech_thread.finished.connect(self.process_speech_result)
        self.speech_thread.error.connect(self.handle_speech_error)
        self.speech_thread.start()

    def select_sign_mode(self):
        """Handle sign language mode selection"""
        self.selected_mode = "sign"
        self.mode_selection_widget.hide()
        self.asl_widget.show()
        self.ask_next()
    
    def select_speech_mode(self):
        """Handle speech mode selection"""
        self.selected_mode = "speech"
        self.mode_selection_widget.hide()
        self.asl_widget.show()
        self.ask_next()

def main():
    app = QApplication(sys.argv)
    window = SmartTaxAdvisor()
    window.show()
    sys.exit(app.exec()) 