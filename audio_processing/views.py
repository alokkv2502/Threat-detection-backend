import os
import json
import wave
from django.conf import settings
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from rapidfuzz import fuzz
import subprocess
from pydub import AudioSegment
import torch
import librosa
from RealtimeSTT import AudioToTextRecorder
import numpy as np
import queue
import threading
import time
import scipy.signal

# Ensure media directory exists
MEDIA_ROOT = os.path.join(settings.BASE_DIR, 'media')
if not os.path.exists(MEDIA_ROOT):
    os.makedirs(MEDIA_ROOT)

# Initialize RealtimeSTT
device = "cuda:0" if torch.cuda.is_available() else "cpu"
realtime_stt = AudioToTextRecorder(
    model="tiny",
    device="cpu",
    compute_type="float32",
    batch_size=1
)

# Define words associated with threats
THREAT_WORDS = ["bomb", "attack", "shoot", "kill", "threat", "danger", "terrorist"]

# Global queue for audio chunks
audio_queue = queue.Queue()
is_listening = False

def record_page(request):
    """ Render the recording UI """
    return render(request, 'record.html')

def process_audio_chunk(chunk):
    """Process a single audio chunk and return transcription and threat status"""
    try:
        # Downsample to 8000 Hz if possible
        if len(chunk) > 0:
            chunk = scipy.signal.resample(chunk, int(len(chunk) * 8000 / 16000))
        # Re-instantiate the model for each chunk (if OOM persists)
        result = AudioToTextRecorder(
            model="tiny",
            device="cpu",
            compute_type="float32",
            batch_size=1
        ).transcribe(chunk)
        if result and result.text:
            is_threat = any(word in result.text.lower() for word in THREAT_WORDS)
            return {
                "text": result.text,
                "is_threat": is_threat,
                "threat_words": [word for word in THREAT_WORDS if word in result.text.lower()]
            }
    except Exception as e:
        print(f"Error processing chunk: {str(e)}")
    return None

def audio_processor():
    """Background thread to process audio chunks"""
    while is_listening:
        try:
            if not audio_queue.empty():
                chunk = audio_queue.get()
                result = process_audio_chunk(chunk)
                if result:
                    yield f"data: {json.dumps(result)}\n\n"
            time.sleep(0.1)  # Small delay to prevent CPU overload
        except Exception as e:
            print(f"Error in audio processor: {str(e)}")

@csrf_exempt
def start_listening(request):
    """Start the real-time listening process"""
    global is_listening
    is_listening = True
    
    def event_stream():
        try:
            for result in audio_processor():
                yield result
        finally:
            is_listening = False
    
    return StreamingHttpResponse(
        event_stream(),
        content_type='text/event-stream'
    )

@csrf_exempt
def stop_listening(request):
    """Stop the real-time listening process"""
    global is_listening
    is_listening = False
    return JsonResponse({"status": "stopped"})

@csrf_exempt
def stream_audio(request):
    """Handle incoming audio stream chunks"""
    if request.method == "POST" and request.FILES.get("audio"):
        try:
            audio_file = request.FILES["audio"]
            # Convert the audio chunk to numpy array
            audio_data = np.frombuffer(audio_file.read(), dtype=np.float32)
            # Add to processing queue
            audio_queue.put(audio_data)
            return JsonResponse({"status": "received"})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    return JsonResponse({"error": "No audio data received"}, status=400)