import essentia.standard as es
import os
import json
import numpy as np
from pathlib import Path


def load_audio(filepath):
    """
    Loads an audio file, returning the original stereo signal, the original mono signal,
    a resampled mono signal, and the original sampling rate.
    - filepath: Path to the audio file.
    Returns a tuple of (resampled_mono_audio, original_mono_audio, stereo_audio, sr).
    """
    # Load the audio file. AudioLoader returns the audio in the same number of channels as the file.
    audio, sr, nc, _, _, _ = es.AudioLoader(filename=str(filepath))()
    assert int(sr) == 44100
    # Check if the audio is already mono
    if nc == 1:
        mono_audio = audio
    else:
        # If the audio is stereo or more, mix down to mono for the mono-specific processing
        mono_audio = es.MonoMixer()(audio, nc)

    # Keep a copy of the original mono audio before resampling
    original_mono_audio = mono_audio

    # Resample the mono audio to 16kHz for consistent processing
    resampled_mono_audio = es.Resample(inputSampleRate=sr, outputSampleRate=16000)(mono_audio)

    # Ensure the stereo audio is returned unchanged if it's originally stereo,
    # or duplicated to stereo if mono
    stereo_audio = audio if nc > 1 else np.tile(audio, (2, 1)).T  # Duplicate mono channel to create stereo if necessary

    return resampled_mono_audio, original_mono_audio, stereo_audio, sr


def find_audio_files(directory):
    """
    Recursively finds all audio files in a directory, filtering by supported extensions.
    - directory: Directory path to search for audio files.
    Returns a list of audio file paths.
    """
    supported_extensions = ['.mp3', '.wav', '.flac', '.aac']
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in supported_extensions:
                audio_files.append(Path(root) / file)
    return audio_files


def save_features(features, filename):
    """
    Saves extracted features to a file in JSON format.
    - features: Dictionary of features to save.
    - filename: Filename for the saved JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(features, f, indent=4)


# TODO
def load_features(filename):
    pass