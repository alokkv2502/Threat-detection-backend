import os
import json
import wave
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from rapidfuzz import fuzz
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import subprocess  # For FFmpeg conversion
from pydub import AudioSegment
import torch
import librosa

from transformers import pipeline
from datasets import load_dataset
# Ensure media directory exists
device = "cuda:0" if torch.cuda.is_available() else "cpu"

whisper_model = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    chunk_length_s=30,
    device=device,
)
MEDIA_ROOT = os.path.join(settings.BASE_DIR, 'media')
if not os.path.exists(MEDIA_ROOT):
    os.makedirs(MEDIA_ROOT)





# Define words associated with threats
THREAT_WORDS = ["bomb", "attack", "shoot", "kill", "threat", "danger", "terrorist"]

def record_page(request):
    """ Render the recording UI """
    return render(request, 'record.html')


def convert_to_wav(audio_path):
    """ Convert uploaded audio file to WAV format """
    output_wav = audio_path.replace(".mp3", ".wav").replace(".m4a", ".wav").replace(".ogg", ".wav")

    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # 16kHz, mono, 16-bit PCM
        audio.export(output_wav, format="wav")
        return output_wav
    except Exception as e:
        print("Conversion Error:", e)
        return None

@csrf_exempt
def upload_audio(request):
    """ Handle file upload from the frontend """
    if request.method == "POST" and request.FILES.get("audio"):
        audio_file = request.FILES["audio"]
        file_path = os.path.join(MEDIA_ROOT, "user_audio.wav")

        # Save the uploaded file
        with open(file_path, "wb+") as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)

        return JsonResponse({"message": "Audio uploaded successfully", "file_path": file_path})
    
    return JsonResponse({"error": "No file received"}, status=400)
@csrf_exempt
def process_audio(request):
    """ Process the uploaded audio using Whisper Large """
    audio_file_path = os.path.join(MEDIA_ROOT, "user_recording.wav")

    if not os.path.exists(audio_file_path):
        return JsonResponse({"error": "No recorded audio found"}, status=400)

    try:
        # Load and preprocess audio with librosa
        audio, sr = librosa.load(audio_file_path, sr=16000)

        # Transcribe using Whisper
        prediction = whisper_model(audio, batch_size=8, return_timestamps=True)

        transcript = prediction["text"]
        timestamp_chunks = prediction["chunks"]

        # Threat word detection (You can customize this)
        THREAT_WORDS = ["attack", "bomb", "kill", "danger", "explode"]
        is_threat = any(word in transcript.lower() for word in THREAT_WORDS)

        return JsonResponse({
            "message": "Processing complete",
            "transcript": transcript,
            "timestamps": timestamp_chunks,  # Optional: Return timestamps for words
            "threat_level": "High" if is_threat else "Low"
        })

    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return JsonResponse({"error": "Processing failed"}, status=500)