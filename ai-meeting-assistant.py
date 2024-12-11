print("Importing libraries ...")
import os
import sys
import torch
import ffmpeg
import numpy as np
from pydub import AudioSegment
print("Importing Hugging Face libraries ...")
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration

def extract_audio_from_video(video_path, output_audio_path="output_audio.wav"):
    print("STEP 1: Audio Extraction")
    print(f"Extracting audio from video: {video_path}")
    ffmpeg.input(video_path).output(output_audio_path).run()
    print(f"Audio extraction for video {video_path} is finished. Audio output saved at: {output_audio_path}")
    return output_audio_path


def split_audio(audio, chunk_length_ms=30000):
    return [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]


def transcribe_audio(audio_path):
    print("STEP 2: Audio Transcription")
    print("Loading the Whisper model from OpenAI")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium")

    print(f"Loading and preprocessing the audio file {audio_path} ...")
    # Load and resample audio
    audio = AudioSegment.from_file(audio_path).set_frame_rate(16000).set_channels(1)

    chunks = split_audio(audio)
    transcriptions = []
    for chunk in chunks:
        audio_samples = np.array(chunk.get_array_of_samples()).astype(np.float32) / 32768.0
        input_features = processor(audio_samples, sampling_rate=16000, return_tensors="pt").input_features
        print(f"Performing transcription ...")
        predicted_ids = model.generate(input_features, max_length=448)
        transcriptions.append(processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])
    full_transcription = " ".join(transcriptions)
    return full_transcription


def summarize_text(transcription):
    print("STEP 3: Text Summarization")

    print(f"Loading summarization pipeline ...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    print(f"Summarizing text ...")

    summary = summarizer(transcription, max_length=150, min_length=30, do_sample=False)
    
    print("Parse summary into bullet points")
    bullet_points = "\n".join([f"- {point.strip()}" for point in summary[0]['summary_text'].split('.') if point])
    return bullet_points


def save_summary_to_file(summary, output_path):
    print("STEP 4: Saving summary to ")
    with open(output_path, "w") as file:
        file.write("Key Takeaways:\n")
        file.write(summary)


def main(video_path):
    # Extract audio from video
    audio_file_path = extract_audio_from_video(video_path)
    
    # Transcribe audio to text
    transcriptions = transcribe_audio(audio_path=audio_file_path)
    
    # Generate summary from transcription
    summary = summarize_text(transcription=transcriptions)
    
    # Save summary to file
    save_summary_to_file(summary, output_path=os.path.split(video_path)[1].replace('mp4', 'txt'))
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ai-meeting-assistant.py <video_file_path>")
    else:
        video_path = sys.argv[1]
        main(video_path)
