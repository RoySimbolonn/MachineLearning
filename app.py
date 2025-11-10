import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="🎵 Music Recommender", page_icon="🎵", layout="wide")

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("song_data_cleaned.csv")
    return df

df = load_data()

st.title("🎵 Music Recommendation System")
st.write("Temukan lagu serupa berdasarkan fitur audio 🎧")

# ==========================================
# PREPROCESSING
# ==========================================
FEATURES = [
    'danceability', 'energy', 'audio_valence',
    'acousticness', 'tempo', 'instrumentalness',
    'speechiness', 'loudness'
]

# Normalisasi fitur
scaler = StandardScaler()
X = scaler.fit_transform(df[FEATURES])

# Fit KNN
model = NearestNeighbors(n_neighbors=6, metric='euclidean')
model.fit(X)

# ==========================================
# INPUT
# ==========================================
song_names = df['song_name'].tolist()
selected_song = st.selectbox("Pilih Lagu:", options=[''] + sorted(song_names))

if selected_song:
    idx = df[df['song_name'] == selected_song].index[0]
    input_features = X[idx].reshape(1, -1)

    distances, indices = model.kneighbors(input_features)
    rec_indices = indices[0][1:]  # skip lagu sendiri

    recommendations = df.iloc[rec_indices][['song_name', 'danceability', 'energy', 'tempo']].reset_index(drop=True)
    st.subheader(f"🎧 Lagu Mirip dengan: {selected_song}")
    st.dataframe(recommendations)

    # Visualisasi radar
    input_values = df.loc[idx, FEATURES].values
    rec_values = df.loc[rec_indices[0], FEATURES].values

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=input_values,
        theta=FEATURES,
        fill='toself',
        name=selected_song
    ))
    fig.add_trace(go.Scatterpolar(
        r=rec_values,
        theta=FEATURES,
        fill='toself',
        name=recommendations.iloc[0]['song_name']
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title="Perbandingan Fitur Audio",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Pilih lagu untuk melihat rekomendasi 🎶")
