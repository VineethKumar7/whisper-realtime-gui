# file-to.text.py
import whisper
import json
from datetime import timedelta

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds // 60) % 60
    seconds = td.seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def transcribe_with_timestamps(audio_path, model="medium"):
    # Load the larger model for improved accuracy
    model = whisper.load_model(model)
    
    # Transcribe with word timestamps and adjust parameters
    result = model.transcribe(
        audio_path, 
        language="de",  # German as input language
        task="translate",  # Translate directly to English
        word_timestamps=True,
        condition_on_previous_text=False,
        no_speech_threshold=0.5,
        logprob_threshold=-1.0
    )
    
    # Format segments
    formatted_segments = []
    for segment in result["segments"]:
        formatted_segment = {
            "start_time": format_timestamp(segment["start"]),
            "end_time": format_timestamp(segment["end"]),
            "text": segment["text"].strip(),  # Transcribed and translated text
        }
        formatted_segments.append(formatted_segment)
    
    # Save to JSON file
    with open("transcription.json", "w", encoding="utf-8") as f:
        json.dump(formatted_segments, f, ensure_ascii=False, indent=2)
    
    # Save as formatted text
    with open("transcription.txt", "w", encoding="utf-8") as f:
        for segment in formatted_segments:
            f.write(f'[{segment["start_time"]} -> {segment["end_time"]}]\n')
            f.write(f'{segment["text"]}\n\n')
    
    return formatted_segments

# Usage
audio_file = "demo.WAV"  # Change to your audio file path
segments = transcribe_with_timestamps(audio_file)

# Print results for verification
for segment in segments:
    print(f'[{segment["start_time"]} -> {segment["end_time"]}]')
    print(segment["text"])
    print()
