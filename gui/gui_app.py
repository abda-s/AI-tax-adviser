# gui/gui_app.py
import tkinter as tk
import cv2
from PIL import Image, ImageTk
from config.config import load_config
from questions import QUESTIONS
from input.asl_detector import ASLDetector
from input.speech_recognizer import SpeechRecognizer
from nlp.answer_normalizer import normalize
from kb.tax_engine import TaxEngine
from tts.tts_engine import TTSEngine

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

        # Video feed label
        self.video_label = tk.Label(root)
        self.video_label.pack()

        # Question label
        self.q_label = tk.Label(root, text='', font=('Helvetica', 16))
        self.q_label.pack(pady=10)

        # Control button
        self.start_btn = tk.Button(root, text='Start Questionnaire', command=self.start)
        self.start_btn.pack(pady=5)

        # Result text box
        self.result_text = tk.Text(root, height=4, font=('Helvetica', 14))
        self.result_text.pack(fill='x', padx=10, pady=10)

        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.current_q = 0
        self.answers = {}

    def start(self):
        self.start_btn.config(state='disabled')
        self.ask_next()

    def ask_next(self):
        if self.current_q < len(QUESTIONS):
            q = QUESTIONS[self.current_q]
            self.q_label.config(text=q['text'])
            self.tts.speak(q['text'])
            self.root.after(1000, self.capture_answer)
        else:
            self.finish()

    def capture_answer(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        # Display frame in GUI
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        q = QUESTIONS[self.current_q]
        if q['type'] == 'yesno':
            raw, _ = self.asl.recognize_letter(frame)
        else:
            raw = self.sr.listen()

        val = normalize(raw, q['type'])
        self.answers[q['id']] = val
        self.current_q += 1
        self.root.after(1500, self.ask_next)

    def finish(self):
        self.cap.release()
        result = self.engine.evaluate(self.answers)
        self.tts.speak(f'Result: {result}')
        self.result_text.insert(tk.END, f'Conclusion: {result}')

if __name__ == '__main__':
    root = tk.Tk()
    app = GUIApp(root)
    root.mainloop()