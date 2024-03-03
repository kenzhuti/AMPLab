import numpy as np
import essentia.standard as es
from utils import load_audio  # Make sure this is correctly pointing to your utils.py
import json


def load_metadata(metadata_path):
    # Load metadata for the Discogs-Effnet model
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata


# Global constants
MODEL_PATH = './model/'

# Embeddings and metadata
DICOGS_EFFNET_EMBED_MODEL = MODEL_PATH + 'discogs-effnet-bs64-1.pb' # music style embedding
DISCOGS_EFFNET_METADATA = MODEL_PATH + "discogs-effnet-bs64-1.json" # music style metadata
MSD_MUSICCNN_EMBED_MODEL = MODEL_PATH + 'msd-musicnn-1.pb' # embedding

# Classifiers
MUSIC_GENRE_MODEL = MODEL_PATH + 'genre_discogs400-discogs-effnet-1.pb'
TEMPO_CNN_MODEL = MODEL_PATH + 'deeptemp-k4-3.pb'
VOICE_INSTRUMENT_MODEL = MODEL_PATH + 'voice_instrumental-discogs-effnet-1.pb'
DANCEABILITY_MODEL = MODEL_PATH + 'danceability-discogs-effnet-1.pb'
AROUSAL_VALENCE_MODEL = MODEL_PATH + 'emomusic-msd-musicnn-2.pb'


class FeatureExtractor:
    def __init__(self):
        """
        Initializes various feature extractors using pre-trained models and DSP techniques.
        """
        self.discogs_effnet_metadata = load_metadata(DISCOGS_EFFNET_METADATA)
        self.bpm_extractor = es.TempoCNN(graphFilename=TEMPO_CNN_MODEL)
        self.loudness_extractor = es.LoudnessEBUR128()

        # Initialize key extractors for different profiles
        self.key_extractor_temperley = es.KeyExtractor(profileType="temperley")
        self.key_extractor_krumhansl = es.KeyExtractor(profileType="krumhansl")
        self.key_extractor_edma = es.KeyExtractor(profileType="edma")

        # Load pre-trained models for embeddings and classifiers
        # General embedding
        self.discogs_effnet_embeddings = es.TensorflowPredictEffnetDiscogs(
            graphFilename= DICOGS_EFFNET_EMBED_MODEL,
            output="PartitionedCall:1",
        )

        # Embeddings for Arousal and valence
        self.msd_music_cnn_embeddings = es.TensorflowPredictMusiCNN(
            graphFilename=MSD_MUSICCNN_EMBED_MODEL, output="model/dense/BiasAdd"
        )

        self.discogs_genre_clf = es.TensorflowPredict2D(
            graphFilename=MUSIC_GENRE_MODEL,
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0",
        )
        self.voice_instrumental_clf = es.TensorflowPredict2D(
            graphFilename=VOICE_INSTRUMENT_MODEL,
            output="model/Softmax",
        )
        self.danceability_clf = es.TensorflowPredict2D(
            graphFilename=DANCEABILITY_MODEL,
            output="model/Softmax",
        )
        self.arousal_valence_clf = es.TensorflowPredict2D(
            graphFilename=AROUSAL_VALENCE_MODEL, output="model/Identity"
        )

    def extract_tempo(self, audio):
        global_tempo, _, _ = self.bpm_extractor(audio)
        return global_tempo

    def extract_key(self, audio, profile):
        if profile == 'temperley':
            return self.key_extractor_temperley(audio)
        elif profile == 'krumhansl':
            return self.key_extractor_krumhansl(audio)
        elif profile == 'edma':
            return self.key_extractor_edma(audio)
        else:
            raise ValueError(f"Profile {profile} is not supported for key extraction.")

    def extract_loudness(self, audio):
        _, _, integrated_loudness, _ = self.loudness_extractor(audio)
        return integrated_loudness

    def get_discogs_efnet_embeddings(self, audio: np.array):
        return self.discogs_effnet_embeddings(audio)

    def get_msd_music_cnn_embeddings(self, audio: np.array):
        return self.msd_music_cnn_embeddings(audio)

    def predict_genre(self, embeddings):
        """
        Returns a dictionary with genre labels and their corresponding averaged activation probabilities.
        """
        # Ensure embeddings are in a 2D array (batch of embeddings) format
        if embeddings.ndim == 1:
            # If embeddings are a 1D array (single embedding), add an extra dimension
            embeddings = np.expand_dims(embeddings, axis=0)

        # Average predictions over all the frames
        averaged_activations = np.mean(self.discogs_genre_clf(embeddings), axis=0)

        # Ensure the number of genres matches the number of activations
        assert len(self.discogs_effnet_metadata["classes"]) == len(averaged_activations)

        # Create a dictionary mapping genre labels to their probabilities
        genre_probabilities = {genre: prob for genre, prob in
                               zip(self.discogs_effnet_metadata["classes"], averaged_activations)}

        return genre_probabilities

    def predict_voice_instrumental(self, embeddings):
        """
        Returns the averaged probability of being voice or instrumental over all the frames
        """
        return np.mean(self.voice_instrumental_clf(embeddings), axis=0)[0] # Assuming it's a binary classification

    def predict_danceability(self, embeddings):
        """
        Returns the averaged probability of danceability over all the frames.
        """
        return np.mean(self.danceability_clf(embeddings), axis=0)[0]

    def predict_valence_arousal(self, embeddings):
        """
        Avaraging over all the frames. The input should be 'music_cnn_embeddings'
        """
        return tuple(np.mean(self.arousal_valence_clf(embeddings), axis=0))

    def extract_features(self, audio_path):
        """
        Extract features from the given audio file path.
        """
        # Load audio using the utility function from utils.py
        resampled_mono_audio, mono_audio, stereo_audio, sr = load_audio(audio_path)
        discogs_efnet_embeddings = self.get_discogs_efnet_embeddings(resampled_mono_audio) # 16KHz sr
        music_cnn_embeddings = self.get_msd_music_cnn_embeddings(resampled_mono_audio) # 16KHz sr
        # Extract features using the methods defined in this class
        features = {
            'audio_path': audio_path.as_posix(),
            'tempo': self.extract_tempo(mono_audio),
            'key_temperley': self.extract_key(mono_audio, 'temperley'),
            'key_krumhansl': self.extract_key(mono_audio, 'krumhansl'),
            'key_edma': self.extract_key(mono_audio, 'edma'),
            'loudness': self.extract_loudness(stereo_audio),
            'voice_instrumental': self.predict_voice_instrumental(discogs_efnet_embeddings),
            'danceability': self.predict_danceability(discogs_efnet_embeddings),
            'arousal_valance': self.predict_valence_arousal(music_cnn_embeddings)
        }

        return features

# The rest of the class would include additional methods for the extraction of other features as required.
