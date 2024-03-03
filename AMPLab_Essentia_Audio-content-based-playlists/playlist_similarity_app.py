import streamlit as st
from app.app_backend import load_embedding_from_pickle, compute_similarity, load_audio_paths
import numpy as np

# 加载嵌入数据和音频路径
discogs_embeddings = load_embedding_from_pickle('embedding/discogs.pkl', average_embeddings=True)
musiccnn_embeddings = load_embedding_from_pickle('embedding/musiccnn.pkl', average_embeddings=True)
audio_paths = load_audio_paths('embedding/embed_indexes.txt')  # 确保此函数已正确定义

# Streamlit UI
st.title('Music Similarity Demo')

# 用户选择查询曲目，展示音频路径名称
track_options = {idx: path for idx, path in enumerate(audio_paths)}
track_index = st.selectbox('Select a query track', list(track_options.keys()), format_func=lambda x: track_options[x])

# 显示查询曲目和计算两种嵌入的最相似曲目
if track_index is not None:
    # 获取最相似的曲目
    similar_tracks_discogs = compute_similarity(track_index, discogs_embeddings, use_cosine=True) # Otherwise dot-product
    similar_tracks_musiccnn = compute_similarity(track_index, musiccnn_embeddings, use_cosine=True)

    st.header('Query Track')
    st.write(audio_paths[track_index])  # 展示查询曲目的路径名称
    st.audio(audio_paths[track_index])  # 展示查询曲目的音频播放器

    # 展示Discogs-Effnet嵌入的最相似曲目
    st.header('Most similar tracks using Discogs-Effnet embeddings:')
    for idx, _ in similar_tracks_discogs:
        st.write(audio_paths[idx])  # 展示相似曲目的路径名称
        st.audio(audio_paths[idx])  # 嵌入音频播放器

    # 展示MSD-MusicCNN嵌入的最相似曲目
    st.header('Most similar tracks using MSD-MusicCNN embeddings:')
    for idx, _ in similar_tracks_musiccnn:
        st.write(audio_paths[idx])  # 展示相似曲目的路径名称
        st.audio(audio_paths[idx])  # 嵌入音频播放器
