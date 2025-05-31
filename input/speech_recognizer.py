# input/speech_recognizer.py
import speech_recognition as sr

class SpeechRecognizer:
    def __init__(self, energy_threshold=1000, pause_threshold=1.5):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold

    def listen(self, timeout=2, phrase_time_limit=60):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except Exception:
            return None