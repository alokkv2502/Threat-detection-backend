from django.urls import path
from .views import record_page, upload_audio, process_audio

urlpatterns = [
    path('', record_page, name='record'),
    path('api/upload/', upload_audio, name='audio-upload'),
    path('api/process/', process_audio, name='process-audio'),
]
