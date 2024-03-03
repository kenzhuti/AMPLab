import streamlit as st
from app.app_backend import load_embedding_from_pickle, compute_similarity, load_audio_paths
import numpy as np

# Load embedded data and audio paths
discogs_embeddings = load_embedding_from_pickle('embedding/discogs.pkl', average_embeddings=True)
musiccnn_embeddings = load_embedding_from_pickle('embedding/musiccnn.pkl', average_embeddings=True)
audio_paths = load_audio_paths('embedding/embed_indexes.txt')  # Make sure this function is properly defined

# Streamlit UI
st.title('Music Similarity Demo')

# User selects a query track and displays the audio path name
track_options = {idx: path for idx, path in enumerate(audio_paths)}
track_index = st.selectbox('Select a query track', list(track_options.keys()), format_func=lambda x: track_options[x])

# Display the query track and calculate the most similar track for both embeds
if track_index is not None:
    # Get the most similar tracks
    similar_tracks_discogs = compute_similarity(track_index, discogs_embeddings, use_cosine=True) # Otherwise dot-product
    similar_tracks_musiccnn = compute_similarity(track_index, musiccnn_embeddings, use_cosine=True)

    st.header('Query Track')
    st.write(audio_paths[track_index])  # Display the path name of the query track
    st.audio(audio_paths[track_index])  # Audio player displaying query tracks

    # Showing the most similar tracks to Discogs-Effnet Embedded
    st.header('Most similar tracks using Discogs-Effnet embeddings:')
    for idx, _ in similar_tracks_discogs:
        st.write(audio_paths[idx])  # Display path names of similar tracks
        st.audio(audio_paths[idx])  # Embedd Audio Player

    # Demonstrate the most similar tracks embedded by MSD-MusicCNN
    st.header('Most similar tracks using MSD-MusicCNN embeddings:')
    for idx, _ in similar_tracks_musiccnn:
        st.write(audio_paths[idx])  # Show path names of similar tracks
        st.audio(audio_paths[idx])  # Embed the audio player
