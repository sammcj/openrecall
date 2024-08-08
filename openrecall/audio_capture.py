import pyaudio
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.silence import split_on_silence
import threading
import queue
import time
import tempfile
import os
import wave
from openrecall.database import insert_entry
from openrecall.nlp import get_embedding
from openrecall.config import appdata_folder

# Initialize Whisper model with a larger, more accurate model
model = WhisperModel("medium.en", device="cpu", compute_type="int8")

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16  # Changed from paFloat32 to paInt16
CHANNELS = 1
RATE = 16000

# Voice activity detection parameters
SILENCE_THRESHOLD = 300  # Adjusted for int16 data
MIN_SILENCE_LENGTH = 0.5  # seconds


class AudioProcessor:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.p = pyaudio.PyAudio()
        self.debug_counter = 0
        self.debug_folder = os.path.join(appdata_folder, "audio_debug")
        os.makedirs(self.debug_folder, exist_ok=True)

    def start_recording(self):
        self.is_recording = True
        self.record_thread = threading.Thread(target=self._record_audio)
        self.process_thread = threading.Thread(target=self._process_audio)
        self.record_thread.daemon = True
        self.process_thread.daemon = True
        self.record_thread.start()
        self.process_thread.start()

    def stop_recording(self):
        self.is_recording = False
        if self.record_thread.is_alive():
            self.record_thread.join()
        if self.process_thread.is_alive():
            self.process_thread.join()

    def _record_audio(self):
        stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        debug_frames = []
        while self.is_recording:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                self.audio_queue.put(data)
                debug_frames.append(data)

                # Save debug audio every 10 seconds
                if len(debug_frames) >= (RATE // CHUNK) * 10:
                    self._save_debug_audio(debug_frames)
                    debug_frames = []
            except OSError as e:
                print(f"Error reading audio stream: {e}")
                time.sleep(0.1)

        stream.stop_stream()
        stream.close()

    def _save_debug_audio(self, frames):
        self.debug_counter += 1
        debug_filename = os.path.join(
            self.debug_folder, f"debug_audio_{self.debug_counter}.wav"
        )
        wf = wave.open(debug_filename, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
        wf.close()
        print(f"Saved debug audio: {debug_filename}")

    def _process_audio(self):
        buffer = []
        silence_start = None

        while self.is_recording or not self.audio_queue.empty():
            try:
                data = self.audio_queue.get(timeout=1)
                buffer.append(data)

                # Convert bytes to numpy array
                np_data = np.frombuffer(data, dtype=np.int16)

                # Check for voice activity
                if np.abs(np_data).mean() < SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                elif silence_start is not None:
                    silence_duration = time.time() - silence_start
                    if silence_duration >= MIN_SILENCE_LENGTH:
                        # Process the audio chunk
                        self._transcribe_audio(b"".join(buffer))
                        buffer = []
                    silence_start = None
            except queue.Empty:
                # If no data for 1 second, process whatever is in the buffer
                if buffer:
                    self._transcribe_audio(b"".join(buffer))
                    buffer = []

        # Process any remaining audio
        if buffer:
            self._transcribe_audio(b"".join(buffer))

    def _transcribe_audio(self, audio_chunk):
        # Convert bytes to numpy array
        np_data = np.frombuffer(audio_chunk, dtype=np.int16)

        # Convert to float32 and normalize
        float_data = np_data.astype(np.float32) / 32768.0

        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
            wf = wave.open(temp_wav_path, "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(RATE)
            wf.writeframes(audio_chunk)
            wf.close()

        try:
            # Transcribe
            segments, info = model.transcribe(temp_wav_path, beam_size=5, language="en")

            for segment in segments:
                text = segment.text.strip()
                if text:  # Only process non-empty transcriptions
                    timestamp = int(time.time())
                    embedding = get_embedding(text)

                    # Insert into database
                    insert_entry(
                        text,
                        timestamp,
                        embedding,
                        "AudioTranscription",
                        f"Transcription at {timestamp}",
                    )
                    print(f"Transcribed: {text}")  # Debug print
        except Exception as e:
            print(f"Transcription error: {e}")
        finally:
            # Clean up the temporary file
            os.unlink(temp_wav_path)


audio_processor = AudioProcessor()


def start_audio_capture():
    audio_processor.start_recording()


def stop_audio_capture():
    audio_processor.stop_recording()
