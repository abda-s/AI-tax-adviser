# tts/tts_engine.py
import subprocess
import os
import platform
import wave
import pyaudio

class TTSEngine:
    def __init__(self, model_path="tts_models/en_US-lessac-medium.onnx",
                 config_path="tts_models/en_US-lessac-medium.onnx.json"):
        
        self.platform = platform.system()
        if self.platform == "Windows":
            self.piper_executable = os.path.join("tts_models", "piper", "piper.exe")
        elif self.platform == "Linux": # For Raspberry Pi
            self.piper_executable = os.path.join("tts_models", "piper", "piper")
            # Ensure the executable has permissions
            if not os.access(self.piper_executable, os.X_OK):
                os.chmod(self.piper_executable, 0o755)
        else:
            raise NotImplementedError("Unsupported operating system for Piper TTS")

        if not os.path.exists(self.piper_executable):
            raise FileNotFoundError(f"Piper executable not found at {self.piper_executable}. Please ensure it's in the correct subfolder (e.g., tts_models/piper/piper.exe).")
        
        self.model_path = model_path
        self.config_path = config_path

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Piper model file not found at {self.model_path}")
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Piper config file not found at {self.config_path}")

    def speak(self, text: str):
        output_wav_path = "temp_output.wav"
        
        command = [
            self.piper_executable,
            "--model", self.model_path,
            "--config", self.config_path,
            "--output_file", output_wav_path
        ]
        
        # Use subprocess.run with text as stdin
        process = subprocess.run(command, input=text.encode('utf-8'), capture_output=True, check=True)
        
        if process.returncode != 0:
            print(f"Piper TTS error: {process.stderr.decode('utf-8')}")
            return

        # Play the generated WAV file
        if os.path.exists(output_wav_path):
            self._play_wav(output_wav_path)
            os.remove(output_wav_path)

    def _play_wav(self, file_path):
        try:
            wf = wave.open(file_path, 'rb')
        except wave.Error as e:
            print(f"Error opening WAV file: {e}")
            return

        p = pyaudio.PyAudio()

        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)

        stream.stop_stream()
        stream.close()
        wf.close()

        p.terminate()