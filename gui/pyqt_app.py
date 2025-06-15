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
        
        # Set window flags for fullscreen on Raspberry Pi
        if os.environ.get("QT_QPA_PLATFORM") == "eglfs":
            self.setWindowFlags(Qt.FramelessWindowHint)
            self.showFullScreen()
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
        self.listening_dots = 0
        self.speech_thread = None
        
        # Debounce settings
        self.FRAME_BUFFER = 5
        self.buffer = deque(maxlen=self.FRAME_BUFFER)
        self.last_label = None

        # Webcam setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        
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
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        
        # Other UI elements
        self.question_label = QLabel()
        self.question_label.setStyleSheet("font-size: 16px;")
        self.status_label = QLabel()
        self.status_label.setStyleSheet("font-size: 12px; color: blue;")
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        
        # Create microphone indicator
        self.mic_indicator = MicrophoneIndicator()
        self.mic_indicator.hide()
        
        # Add widgets to layout
        self.layout.addWidget(self.mode_label)
        self.layout.addLayout(button_layout)
        self.layout.addWidget(self.video_label)
        self.layout.addWidget(self.question_label)
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.mic_indicator, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.result_text)
        
        # Hide video and question widgets initially
        self.video_label.hide()
        self.question_label.hide()
        self.status_label.hide()
        self.result_text.hide()

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
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with ASL detector
            if self.asking_digits:
                label, conf = self.asl.recognize_digit(frame)
            else:
                label, conf = self.asl.recognize_letter(frame)
            
            if label:
                self.handle_sign_detection(label, conf)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error processing sign frame: {e}")
            return frame
    
    def handle_sign_detection(self, label, confidence):
        """Handle detected sign language input"""
        if self.asking_digits:
            if label.isdigit():
                self.current_number += label
                if len(self.current_number) == self.expected_digits:
                    self.answers[self.current_q] = self.current_number
                    self.asking_digits = False
                    self.ask_next()
        else:
            if label.upper() in ['Y', 'N']:
                self.answers[self.current_q] = label.upper()
                self.ask_next()
    
    def select_sign_mode(self):
        """Handle sign language mode selection"""
        self.selected_mode = "sign"
        self.video_label.show()
        self.question_label.show()
        self.status_label.show()
        self.ask_next()
    
    def select_speech_mode(self):
        """Handle speech mode selection"""
        self.selected_mode = "speech"
        self.video_label.show()
        self.question_label.show()
        self.status_label.show()
        self.ask_next()
    
    def ask_next(self):
        """Ask the next question"""
        if self.current_q < len(QUESTIONS):
            question = QUESTIONS[self.current_q]
            self.question_label.setText(question['text'])
            
            if question.get('type') == 'digits':
                self.asking_digits = True
                self.expected_digits = question.get('digits', 1)
                self.current_number = ""
            else:
                self.asking_digits = False
            
            self.current_q += 1
        else:
            self.finish()
    
    def finish(self):
        """Handle completion of questions"""
        self.video_label.hide()
        self.question_label.hide()
        self.status_label.hide()
        
        # Process answers and display results
        result = self.engine.process_answers(self.answers)
        self.result_text.setText(result)
        self.result_text.show()

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

def main():
    app = QApplication(sys.argv)
    window = SmartTaxAdvisor()
    window.show()
    sys.exit(app.exec()) 