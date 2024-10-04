from django.shortcuts import render

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import default_storage
import os
from jiwer import wer, cer
from rest_framework.response import Response
from django.shortcuts import render

# Load the model and processor globally
processor, model = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
), Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

reference_transcription = """A merchant traveled through the forest with his camel. When the camel fell sick, the merchant abandoned it and continued his journey. The camel soon recovered by eating what it found in the forest.

A lion, a fox, and a crow lived together in that forest. They learned about the camel and became friends with it after hearing its story.

One day, the lion was injured while hunting an elephant. Unable to hunt for food, the lion grew hungry and called upon his friends, the fox, the crow, and the camel. 

"Find me food," the lion instructed each animal, sending them in different directions.

The three friends searched but couldn't find any food. They decided to return to the lion and offer themselves as a meal.

The crow volunteered first, but the lion found it too small. The fox offered itself next, but received the same response.

Finally, the camel spoke, "I'm the largest and can satisfy everyone's hunger. You should eat me." 

As soon as the camel offered itself, the lion and the fox pounced on it.

Moral those who are cunning will always find ways to fulfill their evil intentions, no matter how friendly they seem.  Trusting others blindly can lead to unexpected difficulties. Even if someone appears good at first, their true nature will eventually reveal itself.
"""


def index(request):
    return render(request, "index.html")  # Renders the HTML page


def transcribe(audio, processor, model):
    # Check if audio is too short (less than 0.5 seconds)
    if len(audio) < 8000:  # Assuming 16kHz sample rate
        # Pad the audio to 0.5 seconds
        padding = np.zeros(8000 - len(audio))
        audio = np.concatenate([audio, padding])

    input_values = processor(
        audio, return_tensors="pt", sampling_rate=16000
    ).input_values

    # Get the attention mask
    attention_mask = torch.ones(
        input_values.shape, dtype=torch.long, device=input_values.device
    )

    # Forward pass
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    print(transcription)
    return transcription


@api_view(["POST"])
def transcribe_audio(request):
    if "audio" not in request.FILES:
        return Response({"error": "No audio file provided"}, status=400)

    audio_file = request.FILES["audio"]
    file_name = default_storage.save(audio_file.name, audio_file)
    file_path = default_storage.path(file_name)

    # Read the audio file
    audio, sample_rate = sf.read(file_path)

    # Ensure audio is mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample to 16kHz if necessary
    if sample_rate != 16000:
        audio = np.interp(
            np.linspace(0, len(audio), int(len(audio) * 16000 / sample_rate)),
            np.arange(len(audio)),
            audio,
        )

    # Perform transcription
    result = transcribe(audio, processor, model)

    # Calculate WER and CER if ground truth transcription is provided
    wer_score = wer(reference_transcription, result)
    cer_score = cer(reference_transcription, result)
    metrics = {"wer": wer_score, "cer": cer_score}

    # Clean up the uploaded file
    os.remove(file_path)

    print(metrics)

    return Response({"transcription": result})
