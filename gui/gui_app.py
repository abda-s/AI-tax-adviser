# gui/gui_app.py
import tkinter as tk
import cv2
from PIL import Image, ImageTk
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

class GUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Smart Tax Advisor')

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
        self.text_processor = TextProcessor()  # Initialize text processor

        # Mode selection frame
        self.mode_frame = tk.Frame(root)
        self.mode_frame.pack(pady=20)
        
        # Mode selection label
        self.mode_label = tk.Label(self.mode_frame, text='Select Input Mode:', font=('Helvetica', 16))
        self.mode_label.pack(pady=10)
        
        # Mode selection buttons
        self.sign_btn = tk.Button(self.mode_frame, text='Sign Language', command=lambda: self.select_mode('sign'), 
                                 font=('Helvetica', 12), width=15)
        self.sign_btn.pack(pady=5)
        
        self.speech_btn = tk.Button(self.mode_frame, text='Speech Recognition', command=lambda: self.select_mode('speech'),
                                   font=('Helvetica', 12), width=15)
        self.speech_btn.pack(pady=5)

        # Video feed label (initially hidden)
        self.video_label = tk.Label(root)
        
        # Question label
        self.q_label = tk.Label(root, text='', font=('Helvetica', 16))
        
        # Status label for feedback
        self.status_label = tk.Label(root, text='', font=('Helvetica', 12), fg='blue')
        
        # Result text box
        self.result_text = tk.Text(root, height=4, font=('Helvetica', 14))

        # Webcam
        self.cap = cv2.VideoCapture(0)
        # Set camera resolution to a square format to fix MediaPipe warning
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        
        self.current_q = 0
        self.answers = {}
        self.selected_mode = None
        self.current_number = ""  # Store the current number being built
        self.expected_digits = 0  # Number of digits expected for the current question
        self.no_hand_frames = 0  # Count frames with no hand detected
        self.can_accept_digit = True  # Flag to control when we can accept a new digit
        self.asking_digits = False  # Flag to track if we're asking for number of digits
        self.is_listening = False  # Flag to track if we're currently listening for speech

        # Debounce settings
        self.FRAME_BUFFER = 5
        self.buffer = deque(maxlen=self.FRAME_BUFFER)
        self.last_label = None
        self.is_capturing = False

    def ask_next(self):
        if self.current_q < len(QUESTIONS):
            q = QUESTIONS[self.current_q]
            self.q_label.config(text=q['text'])
            self.tts.speak(q['text'])
            self.current_number = ""  # Reset current number for new question
            self.no_hand_frames = 0  # Reset no hand frames counter
            self.last_label = None  # Reset last label for yes/no questions
            self.can_accept_digit = True  # Reset digit acceptance flag
            self.asking_digits = False  # Reset asking digits flag
            
            # Set expected number of digits based on question
            if q['type'] == 'number':
                if 'children' in q['text'].lower():
                    self.expected_digits = 1  # Single digit for number of children
                else:
                    # For salary, first ask how many digits (only in ASL mode)
                    if self.selected_mode == 'sign':
                        self.asking_digits = True
                        self.q_label.config(text="How many digits is your salary? Show a number between 1 and 5")
                        self.tts.speak("How many digits is your salary? Show a number between 1 and 5")
                    else:
                        # In speech mode, just ask for the salary directly
                        self.expected_digits = 5  # Allow up to 5 digits for salary
            
            if self.selected_mode == 'sign':
                self.is_capturing = True
                self.update_frame()
            else:
                self.capture_speech_answer()
        else:
            self.finish()

    def update_status(self, message, color='blue'):
        self.status_label.config(text=message, fg=color)
        self.root.update()

    def capture_speech_answer(self):
        if self.is_listening:
            return

        self.is_listening = True
        self.update_status("Listening...", "green")
        
        try:
            q = QUESTIONS[self.current_q]
            raw = self.sr.listen()
            
            if raw:
                # Show the raw recognized text
                self.update_status(f"Raw text: {raw}", "blue")
                self.result_text.delete(1.0, tk.END)  # Clear previous text
                self.result_text.insert(tk.END, f"Raw text: {raw}\n")
                
                # Process the text using Gemini
                processed_text = self.text_processor.process_text(raw, q['type'])
                self.result_text.insert(tk.END, f"Processed text: {processed_text}\n")
                
                # For speech mode, accept any valid number for salary
                if q['type'] == 'number' and 'salary' in q['text'].lower():
                    if processed_text.isdigit() and 1 <= int(processed_text) <= 99999:  # Allow up to 5 digits
                        self.answers[q['id']] = processed_text
                        self.current_q += 1
                        self.is_listening = False
                        self.update_status("")
                        self.root.after(1500, self.ask_next)
                    else:
                        self.update_status("Please say a valid number between 1 and 99999", "red")
                        self.result_text.insert(tk.END, "Please say a valid number between 1 and 99999")
                        self.is_listening = False
                        self.root.after(1000, self.capture_speech_answer)
                else:
                    self.answers[q['id']] = processed_text
                    self.current_q += 1
                    self.is_listening = False
                    self.update_status("")
                    self.root.after(1500, self.ask_next)
            else:
                self.update_status("No speech detected. Please try again.", "red")
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "No speech detected. Please try again.")
                self.is_listening = False
                self.root.after(1000, self.capture_speech_answer)
        except sr.WaitTimeoutError:
            self.update_status("No speech detected. Please try again.", "red")
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "No speech detected. Please try again.")
            self.is_listening = False
            self.root.after(1000, self.capture_speech_answer)
        except Exception as e:
            self.update_status(f"Error: {str(e)}", "red")
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error: {str(e)}")
            self.is_listening = False
            self.root.after(1000, self.capture_speech_answer)

    def select_mode(self, mode):
        self.selected_mode = mode
        # Hide mode selection
        self.mode_frame.pack_forget()
        
        # Show video feed if sign language mode
        if mode == 'sign':
            self.video_label.pack()
        
        # Show question label and start questionnaire
        self.q_label.pack(pady=10)
        self.status_label.pack(pady=5)
        self.result_text.pack(fill='x', padx=10, pady=10)
        self.ask_next()

    def update_frame(self):
        if not self.is_capturing:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        q = QUESTIONS[self.current_q]

        # Use sign language for all questions in sign mode
        if self.selected_mode == 'sign':
            if q['type'] == 'yesno':
                label, conf = self.asl.recognize_letter(frame)
                
                # Only accept Y or N for yes/no questions
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

                # Check for stable detection
                if len(self.buffer) == self.FRAME_BUFFER:
                    if self.buffer.count(self.buffer[0]) == self.FRAME_BUFFER:
                        stable = self.buffer[0]
                        if stable and stable != self.last_label:
                            self.last_label = stable
                            val = normalize(stable, q['type'])
                            self.answers[q['id']] = val
                            self.current_q += 1
                            self.is_capturing = False
                            self.root.after(1500, self.ask_next)
                            return
            else:  # For numerical questions
                label, conf = self.asl.recognize_digit(frame)
                
                # Draw prediction on frame
                if label and conf >= self.asl.conf_threshold:
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
                    # If we've had enough frames with no hand, allow accepting a new digit
                    if self.no_hand_frames >= 10:
                        self.can_accept_digit = True

                # Check for stable detection
                if len(self.buffer) == self.FRAME_BUFFER:
                    if self.buffer.count(self.buffer[0]) == self.FRAME_BUFFER:
                        stable = self.buffer[0]
                        if stable and self.can_accept_digit:
                            if self.asking_digits:
                                # For salary, first get the number of digits
                                if '1' <= stable <= '5':
                                    self.expected_digits = int(stable)
                                    self.asking_digits = False
                                    self.q_label.config(text=q['text'])
                                    self.tts.speak(q['text'])
                                    self.can_accept_digit = False
                            else:
                                self.current_number += stable
                                self.can_accept_digit = False  # Prevent immediate re-entry of the same digit
                                # If we have enough digits, process the answer
                                if len(self.current_number) >= self.expected_digits:
                                    val = normalize(self.current_number, q['type'])
                                    self.answers[q['id']] = val
                                    self.current_q += 1
                                    self.is_capturing = False
                                    self.root.after(1500, self.ask_next)
                                    return

        # Display frame in GUI
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        # Schedule next update
        self.root.after(10, self.update_frame)

    def finish(self):
        self.cap.release()
        result = self.engine.evaluate(self.answers)
        self.tts.speak(f'Result: {result}')
        self.result_text.delete(1.0, tk.END)  # Clear previous text
        self.result_text.insert(tk.END, f'Conclusion: {result}')
        self.update_status("Questionnaire completed!", "green")

if __name__ == '__main__':
    root = tk.Tk()
    app = GUIApp(root)
    root.mainloop()