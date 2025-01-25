import pyaudio
import numpy as np
import whisper
import speech_recognition as sr

# Parameters
CHUNK = 1024              # Samples per frame
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 2              # Stereo audio for system audio
RATE = 44100              # Sampling rate for system audio
WHISPER_RATE = 16000      # Sampling rate required by Whisper
BUFFER_DURATION = 5       # Process last 5 seconds of audio

# Initialize Whisper Model
model = whisper.load_model("base")  # Load Whisper model

# Initialize SpeechRecognition for noise reduction
recognizer = sr.Recognizer()

# Rolling buffer for audio data
rolling_buffer = np.zeros(int(WHISPER_RATE * BUFFER_DURATION), dtype=np.float32)

# Get system audio input device index
p = pyaudio.PyAudio()
dev_index = None
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev['name'] == 'Stereo Mix (Realtek(R) Audio)' and dev['hostApi'] == 0:
        dev_index = dev['index']
        print('Using system audio device:', dev['name'])
        break

if dev_index is None:
    raise ValueError("System audio device ('Stereo Mix') not found. Make sure it is enabled and available.")

# Callback function to capture system audio
def audio_callback(in_data, frame_count, time_info, status):
    global rolling_buffer
    # Convert raw audio to NumPy array
    audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
    # Downsample stereo system audio to Whisper's required rate if necessary
    if RATE != WHISPER_RATE:
        audio_data = np.mean(audio_data.reshape(-1, RATE // WHISPER_RATE), axis=1)
    # Add new audio to rolling buffer
    rolling_buffer = np.roll(rolling_buffer, -len(audio_data))
    rolling_buffer[-len(audio_data):] = audio_data
    return (None, pyaudio.paContinue)

# Initialize PyAudio with system audio input
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=dev_index,
    frames_per_buffer=CHUNK,
    stream_callback=audio_callback
)

# Start the audio stream
stream.start_stream()

try:
    print("Listening for Japanese audio from system audio...")

    while True:
        # Normalize rolling buffer audio
        rolling_buffer_normalized = rolling_buffer / np.max(np.abs(rolling_buffer) + 1e-10)

        # Preprocess audio using SpeechRecognition for noise reduction
        audio_data = sr.AudioData(rolling_buffer_normalized.tobytes(), WHISPER_RATE, 2)
        try:
            clean_audio = recognizer.recognize_google(audio_data, language="ja")
            print("SpeechRecognition (Cleaned):", clean_audio)
        except sr.UnknownValueError:
            print("SpeechRecognition could not understand the audio.")
        except sr.RequestError as e:
            print(f"SpeechRecognition API error: {e}")

        # Transcribe audio using Whisper
        print("Processing audio with Whisper...")
        result = model.transcribe(rolling_buffer_normalized, language="ja")
        print("Whisper Transcription:", result["text"])

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
