import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def filter_by_tempo(features, tempo_range):
    """
    Filter tracks by tempo from a list of dictionaries.
    """
    filtered = [track for track in features if tempo_range[0] <= track['tempo'] <= tempo_range[1]]
    return filtered
def filter_by_voice_presence(features, voice_presence):
    """
    Filter tracks by vocal/instrumental presence from a list of dictionaries.
    """
    if voice_presence == 'Vocal':
        return [track for track in features if track['voice_instrumental'] > 0.5]
    elif voice_presence == 'Instrumental':
        return [track for track in features if track['voice_instrumental'] <= 0.5]
    return features

def filter_by_danceability(features, danceability_range):
    """
    Filter tracks by danceability from a list of dictionaries.
    """
    return [track for track in features if danceability_range[0] <= track['danceability'] <= danceability_range[1]]

def filter_by_arousal_valence(features, arousal_range, valence_range):
    """
    Filter tracks by arousal and valence from a list of dictionaries.
    """
    return [track for track in features if arousal_range[0] <= track['arousal_valance'][0] <= arousal_range[1] and
            valence_range[0] <= track['arousal_valance'][1] <= valence_range[1]]

def filter_by_key_scale(features, selected_keys):
    """
    Filter tracks by key/scale from a list of dictionaries, allowing for multiple keys.
    """
    if not selected_keys:  # If no keys are selected, return all features
        return features
    return [track for track in features if any(key.lower() in f"{track['key_temperley'][0]} {track['key_temperley'][1]}".lower() for key in selected_keys)]

def apply_filters(features, tempo_range, voice_presence, danceability_range, arousal_range, valence_range, key):
    """
    Apply all filters to the tracks from a list of dictionaries.
    """
    filtered = filter_by_tempo(features, tempo_range)
    filtered = filter_by_voice_presence(filtered, voice_presence)
    filtered = filter_by_danceability(filtered, danceability_range)
    filtered = filter_by_arousal_valence(filtered, arousal_range, valence_range)
    filtered = filter_by_key_scale(filtered, key)
    return filtered

def load_embedding_from_pickle(file_path, average_embeddings=False):
    all_embeddings = []
    with open(file_path, 'rb') as f:
        while True:
            try:
                embedding = pickle.load(f)  # Assuming that each embedding is a NumPy array of shape (number of time slices, embedding dimension)
                if average_embeddings:
                    # Compute the average embedding on the time slice
                    embedding = np.mean(embedding, axis=0)
                all_embeddings.append(embedding)
            except EOFError:
                break
    return all_embeddings


def compute_similarity(query_idx, embeddings, use_cosine=True):
    similarities = []
    query_embedding = embeddings[query_idx]
    for idx, emb in enumerate(embeddings):
        if idx == query_idx:
            # Skip query embedding itself
            continue

        if use_cosine:
            # Use cosine similarity
            sim = cosine_similarity(query_embedding.reshape(1, -1), emb.reshape(1, -1))[0][0]
        else:
            # Use dot product similarity, assuming vectors are normalised
            query_emb_norm = query_embedding / np.linalg.norm(query_embedding)
            emb_norm = emb / np.linalg.norm(emb)
            sim = np.dot(query_emb_norm, emb_norm)
        similarities.append((idx, sim))
    # Sort by similarity and return an index of the top 10 most similar audio files
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:10]


# load indexed audio paths
def load_audio_paths(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]
