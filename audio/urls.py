from django.contrib import admin
from django.urls import path
from audio_processing.views import record_page, upload_audio, process_audio  # Import your views

urlpatterns = [
    path("admin/", admin.site.urls),  # Django admin panel
    path("", record_page, name="home"),  # Default recording page
    path("upload/", upload_audio, name="upload-audio"),  # Upload audio file
    path("process/", process_audio, name="process-audio"),  # Process audio with ML
]
