import streamlit as st
import pandas as pd
from app.app_backend import apply_filters

features = pd.read_pickle('feature/extracted_features_without_genre.pkl')

# Initialise or update current_page in session_state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1

# UI components for input
st.sidebar.header('Filter Criteria')

# Tempo
tempo_min, tempo_max = st.sidebar.slider('Tempo range (BPM)', min_value=50, max_value=200, value=(80, 120))

# Vocal/Instrumental
voice_presence = st.sidebar.radio('Voice presence', ['Both', 'Vocal', 'Instrumental'])

# Danceability
danceability_range = st.sidebar.slider('Danceability range', 0.0, 1.0, (0.2, 0.8))

# Arousal and Valence
arousal_range = st.sidebar.slider('Arousal range', 1, 9, (1, 9))
valence_range = st.sidebar.slider('Valence range', 1, 9, (1, 9))

# Key/Scale
key_temperley_values = {f"{track['key_temperley'][0]} {track['key_temperley'][1]}" for track in features}
key_temperley_list = sorted(list(key_temperley_values))
selected_keys = st.sidebar.multiselect('Select Key(s)', key_temperley_list)

# Filter Tracks button
if st.sidebar.button('Filter Tracks'):
    filtered_tracks = apply_filters(features, (tempo_min, tempo_max), voice_presence, danceability_range, arousal_range, valence_range, selected_keys)
    st.session_state['filtered_tracks'] = filtered_tracks
    st.session_state['current_page'] = 1  # Reset to page 1 after new filter

total_tracks = len(st.session_state.get('filtered_tracks', []))
tracks_per_page = 10
total_pages = max(1, total_tracks // tracks_per_page + (0 if total_tracks % tracks_per_page == 0 else 1))

# pagination control
col1, col2, col3 = st.columns(3)

with col1:
    if st.button('Previous'):
        st.session_state.current_page = max(1, st.session_state.current_page - 1)
with col2:
    selected_page = st.selectbox('Go to page', list(range(1, total_pages + 1)), index=st.session_state.current_page - 1)
    st.session_state.current_page = selected_page
with col3:
    if st.button('Next'):
        st.session_state.current_page = min(total_pages, st.session_state.current_page + 1)

# Display the tracks of the current page
start_idx = (st.session_state.current_page - 1) * tracks_per_page
end_idx = start_idx + tracks_per_page

for track in st.session_state.get('filtered_tracks', [])[start_idx:end_idx]:
    track_name = track.get('audio_path', 'Unknown Track')
    audio_path = track['audio_path']

    # Display track name
    st.write(f"Track: {track_name}")

    # Audio player showing tracks
    try:
        with open(audio_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/mp3')
    except Exception as e:
        st.error(f"Error loading audio file: {e}")

# Save to Playlist button
if st.button('Save to Playlist'):
    if st.session_state['filtered_tracks']:
        m3u_filepaths_file = 'app/playlist.m3u8'
        with open(m3u_filepaths_file, 'w') as file:
            file.write('#EXTM3U\n')
            for track in st.session_state['filtered_tracks']:
                audio_path = track['audio_path']
                file.write(f'{audio_path}\n')
        st.success(f"Filtered tracks have been saved to {m3u_filepaths_file}.")
    else:
        st.error("No filtered tracks to save. Please filter tracks first.")
