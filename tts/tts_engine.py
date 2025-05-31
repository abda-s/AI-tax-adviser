# tts/tts_engine.py
import pyttsx3

class TTSEngine:
    def __init__(self, rate=150, volume=1.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)

    def speak(self, text: str):
        self.engine.say(text)
        self.engine.runAndWait()