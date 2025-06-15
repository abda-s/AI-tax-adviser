from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor
import cv2
from PIL import Image
import numpy as np
import mediapipe as mp
import logging
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

# Set Qt platform for Linux
os.environ["QT_QPA_PLATFORM"] = "eglfs"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
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
            painter.setBrush(Qt.BrushStyle.NoBrush)
            
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
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Smart Tax Advisor')
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
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
        self.listening_dots = 0
        self.speech_thread = None

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

        # Debounce settings
        self.FRAME_BUFFER = 5
        self.buffer = deque(maxlen=self.FRAME_BUFFER)
        self.last_label = None

        # Camera setup for Linux
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                # Try alternative camera index
                self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
                
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Set buffer size to minimize latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            self.cap = None

        # Setup UI
        self.setup_ui()

        # Create timers
        self.listening_timer = QTimer(self)
        self.listening_timer.timeout.connect(self.update_listening_animation)
        self.mic_animation_timer = QTimer(self)
        self.mic_animation_timer.timeout.connect(self.mic_indicator.update_animation)
        self.mic_animation_timer.start(50)  # Update every 50ms for smooth animation
        
        # Set initial status for speech mode
        self.update_status("Not Listening", "gray")
        self.mic_indicator.set_listening(False)

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
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        
        # Initially hide ASL widget and results widget
        self.asl_widget.hide()
        self.results_widget.hide()

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

    def select_sign_mode(self):
        if not self.cap or not self.cap.isOpened():
            self.status_label.setText("Error: Camera not available")
            self.status_label.setStyleSheet("font-size: 12px; color: red;")
            return
            
        self.selected_mode = 'sign'
        self.mode_label.hide()
        self.sign_btn.hide()
        self.speech_btn.hide()
        self.video_label.show()
        self.question_label.show()
        self.status_label.show()
        self.result_text.show()
        self.mic_indicator.hide()
        self.ask_next()

    def select_speech_mode(self):
        self.selected_mode = 'speech'
        self.mode_label.hide()
        self.sign_btn.hide()
        self.speech_btn.hide()
        self.question_label.show()
        self.status_label.show()
        self.result_text.show()
        self.mic_indicator.show()
        self.update_status("Not Listening", "gray")
        self.mic_indicator.set_listening(False)
        self.ask_next()

    def ask_next(self):
        if self.current_q < len(QUESTIONS):
            q = QUESTIONS[self.current_q]

            # Skip marriage-related questions if not married
            if q['id'] in [1, 2, 3, 4]:  # Questions about children, wife's work, salary, and joint filing
                if self.answers.get(0) == 'no':  # If not married
                    # Set default values for skipped questions
                    self.answers[1] = '0'   # No children
                    self.answers[2] = 'no'  # Wife doesn't work
                    self.answers[3] = '0'   # Wife's salary is 0
                    self.answers[4] = 'no'  # Not filing jointly (can't file jointly if not married)
                    self.current_q = 5      # Skip to salary question
                    self.ask_next()         # Recursively call to ask the next question
                    return

            # For joint filing question, ensure user is married
            if q['id'] == 4:  # Joint filing question
                if self.answers.get(0) != 'yes':  # If not married
                    self.answers[4] = 'no'  # Can't file jointly if not married
                    self.current_q += 1
                    self.ask_next()
                    return
                else:
                    # If married, ask about joint filing
                    self.question_label.setText("Are you filing taxes jointly with your spouse?")
                    self.tts.speak("Are you filing taxes jointly with your spouse?")

            # Conditional logic for spouse salary question (Q3)
            if q['id'] == 3:
                # Check answer to Q2 ('Does your wife work?')
                wife_works_answer = self.answers.get(2)
                if wife_works_answer == 'no':
                    # If wife doesn't work, set salary to 0 and skip this question
                    self.answers[q['id']] = '0'
                    self.current_q += 1
                    self.ask_next() # Recursively call to ask the next question
                    return # Exit to prevent asking the current question

            # For other questions, proceed normally
            if q['id'] != 4:  # Don't set text for joint filing question as it's handled above
                self.question_label.setText(q['text'])
                self.tts.speak(q['text'])
            
            self.current_number = ""
            self.no_hand_frames = 0
            self.last_label = None
            self.can_accept_digit = True
            self.asking_digits = False
            
            if q['type'] == 'number':
                if 'children' in q['text'].lower():
                    self.expected_digits = 1
                else:
                    if self.selected_mode == 'sign':
                        self.asking_digits = True
                        self.question_label.setText("How many digits is your salary? Show a number between 1 and 5")
                        self.tts.speak("How many digits is your salary? Show a number between 1 and 5")
                    else:
                        self.expected_digits = 5
            
            if self.selected_mode == 'sign':
                self.is_capturing = True
                self.update_frame()
            else:
                self.capture_speech_answer()
        else:
            self.finish()

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
        
        self.mic_indicator.set_listening(False)
        self.update_status("Not Listening", "gray")

    def handle_speech_error(self, error_msg):
        """Handle speech recognition errors"""
        self.listening_timer.stop()
        self.mic_indicator.set_listening(False)
        self.update_status("Not Listening", "gray")
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

    def update_frame(self):
        if not self.is_capturing or not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        q = QUESTIONS[self.current_q]

        if self.selected_mode == 'sign':
            if q['type'] == 'yesno':
                label, conf = self.asl.recognize_letter(frame)
                
                if label and label.upper() in ['Y', 'N'] and conf >= self.asl.conf_threshold:
                    cv2.putText(frame, f'{label} ({conf*100:.1f}%)', (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    self.buffer.append(label.upper())
                    self.no_hand_frames = 0
                else:
                    cv2.putText(frame, 'Show Y or N sign', (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.buffer.append(None)
                    self.no_hand_frames += 1

                if len(self.buffer) == self.FRAME_BUFFER:
                    if self.buffer.count(self.buffer[0]) == self.FRAME_BUFFER:
                        stable = self.buffer[0]
                        if stable and stable != self.last_label:
                            self.last_label = stable
                            val = normalize(stable, q['type'])
                            self.answers[q['id']] = val
                            self.current_q += 1
                            self.is_capturing = False
                            QTimer.singleShot(1500, self.ask_next)
                            return
            else:
                label, conf = self.asl.recognize_digit(frame)
                
                # Check for hand removal
                if label is None:
                    self.hand_removed_frames += 1
                    if self.hand_removed_frames >= self.hand_removed_threshold:
                        self.hand_removed = True
                        self.hand_removed_frames = 0
                else:
                    self.hand_removed_frames = 0
                    self.hand_removed = False
                
                if label and conf >= self.asl.conf_threshold and self.hand_removed:
                    cv2.putText(frame, f'Current: {self.current_number}{label} ({conf*100:.1f}%)', (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    self.buffer.append(label)
                    self.no_hand_frames = 0
                else:
                    if self.asking_digits:
                        cv2.putText(frame, 'Show number of digits (1-5)', (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, f'Current: {self.current_number} ({len(self.current_number)}/{self.expected_digits})', (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.buffer.append(None)
                    self.no_hand_frames += 1
                    if self.no_hand_frames >= 10:
                        self.can_accept_digit = True

                if len(self.buffer) == self.FRAME_BUFFER:
                    if self.buffer.count(self.buffer[0]) == self.FRAME_BUFFER:
                        stable = self.buffer[0]
                        if stable and self.can_accept_digit and self.hand_removed:
                            if self.asking_digits:
                                if '1' <= stable <= '5':
                                    self.expected_digits = int(stable)
                                    self.asking_digits = False
                                    self.question_label.setText(q['text'])
                                    self.tts.speak(q['text'])
                                    self.can_accept_digit = False
                            else:
                                self.current_number += stable
                                self.can_accept_digit = False
                                if len(self.current_number) >= self.expected_digits:
                                    val = normalize(self.current_number, q['type'])
                                    self.answers[q['id']] = val
                                    self.current_q += 1
                                    self.is_capturing = False
                                    QTimer.singleShot(1500, self.ask_next)
                                    return

        # Convert frame to Qt image and display
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

        # Schedule next update
        QTimer.singleShot(10, self.update_frame)

    def finish(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        # Display collected answers
        answers_summary = """<h2>Your Answers:</h2>"""
        for q_id, answer in self.answers.items():
            question_text = QUESTIONS[q_id]['text']
            answers_summary += f"<b>{question_text}</b> {answer}<br>"
        
        self.result_text.setHtml(answers_summary)
        
        # Prepare facts for TaxEngine
        processed_facts = {
            'married': self.answers.get(0) == 'yes',
            'children': int(self.answers.get(1, '0')),
            'wife_income': int(self.answers.get(3, '0')) > 0,
            'joint_filing': self.answers.get(4) == 'yes'
        }

        # Get and display the final result
        result = self.engine.evaluate(processed_facts)
        self.tts.speak(f'Result: {result}')
        self.result_text.append(f"<br><h2>Conclusion: {result}</h2>")
        self.update_status("Questionnaire completed!", "green")

def main():
    app = QApplication(sys.argv)
    window = SmartTaxAdvisor()
    window.show()
    sys.exit(app.exec()) 