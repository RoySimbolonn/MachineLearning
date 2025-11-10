import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# Judul Aplikasi
st.title("🎵 Music Recommendation App (KNN Model)")
st.write("Masukkan nilai fitur lagu untuk menemukan lagu yang mirip.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("song_data_knn_features.csv")
    return df

df = load_data()

# Fitur yang digunakan
FEATURES = [
    'danceability',
    'energy',
    'audio_valence',
    'acousticness',
    'tempo',
    'instrumentalness',
    'speechiness',
    'loudness'
]

# Normalisasi data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[FEATURES])

# Latih model KNN
model = NearestNeighbors(n_neighbors=5)
model.fit(df_scaled)

# Input pengguna
st.sidebar.header("Input Fitur Lagu")
input_data = []
for f in FEATURES:
    val = st.sidebar.slider(f, float(df[f].min()), float(df[f].max()), float(df[f].mean()))
    input_data.append(val)

# Prediksi lagu mirip
distances, indices = model.kneighbors([input_data])

# Tampilkan hasil
st.subheader("🎧 Lagu yang Mirip:")
st.dataframe(df.iloc[indices[0]])

st.write("---")
st.write("Aplikasi ini menggunakan algoritma **KNN** untuk merekomendasikan lagu berdasarkan kemiripan fitur audio.")
