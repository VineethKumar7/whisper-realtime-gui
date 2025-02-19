# realtime_whisper.py
import sounddevice as sd
import numpy as np
import whisper
import queue
import threading
import time
from datetime import datetime

# Load Whisper model (use at least 'medium' for translation)
model = whisper.load_model("medium")

# Audio capture settings
samplerate = 16000  # Whisper requires 16kHz
channels = 1
blocksize = 30 * samplerate  # Buffer for every 30 seconds
audio_queue = queue.Queue()
recording = True

def audio_callback(indata, frames, time, status):
    """Callback function to capture audio"""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def process_audio():
    """Process audio and transcribe in real time"""
    while recording:
        if not audio_queue.empty():
            # Get audio data from queue
            audio_data = audio_queue.get()
            
            # Flatten the audio for Whisper
            audio_data = audio_data.flatten().astype(np.float32)
            
            try:
                # Transcribe German speech and translate it into English
                result = model.transcribe(audio_data, language="de", task="translate")
                
                # Print results with timestamps
                if result["text"].strip():
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] {result['text']}")  # Translated English text
            except Exception as e:
                print(f"Error during recognition: {str(e)}")

def main():
    global recording
    
    print("Starting real-time German-to-English transcription... (Press Ctrl+C to stop)")
    
    # Start audio processing thread
    process_thread = threading.Thread(target=process_audio)
    process_thread.start()
    
    try:
        # Start recording
        with sd.InputStream(samplerate=samplerate,
                            channels=channels,
                            callback=audio_callback,
                            blocksize=blocksize):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping transcription...")
        recording = False
        process_thread.join()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

