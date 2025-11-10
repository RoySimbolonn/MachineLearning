import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# -------------------------------
# 1️⃣ Load dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("song_data_cleaned.csv")  # pastikan nama file sesuai
    return df

df = load_data()

# -------------------------------
# 2️⃣ Cek kolom yang tersedia
# -------------------------------
st.title("🎵 Music Recommendation App")
st.write("Temukan lagu serupa berdasarkan kemiripan fitur audio.")

# Coba deteksi nama kolom lagu
possible_title_cols = ['song_name', 'title', 'name', 'track_name']
title_col = next((c for c in possible_title_cols if c in df.columns), None)

if title_col is None:
    st.error("❌ Kolom nama lagu tidak ditemukan di dataset. Kolom yang tersedia:")
    st.write(df.columns.tolist())
    st.stop()

# Daftar fitur yang digunakan (pastikan semua ada di dataset)
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
missing_feats = [f for f in FEATURES if f not in df.columns]
if missing_feats:
    st.error(f"❌ Kolom fitur berikut tidak ditemukan di dataset: {missing_feats}")
    st.write("Kolom yang tersedia:", df.columns.tolist())
    st.stop()

# -------------------------------
# 3️⃣ Normalisasi fitur
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURES])

# Buat model KNN
model = NearestNeighbors(n_neighbors=6, metric='euclidean')
model.fit(X_scaled)

# -------------------------------
# 4️⃣ Input pencarian lagu
# -------------------------------
search_query = st.text_input("🔍 Cari lagu yang kamu suka:")

if search_query:
    # Cari lagu yang paling mirip dengan query
    matches = df[df[title_col].str.contains(search_query, case=False, na=False)]

    if len(matches) == 0:
        st.warning("🚫 Lagu tidak ditemukan. Coba kata kunci lain.")
    else:
        st.success(f"🎧 Ditemukan {len(matches)} lagu yang cocok.")
        selected_song = st.selectbox("Pilih salah satu:", matches[title_col].tolist())

        # -------------------------------
        # 5️⃣ Rekomendasi Lagu
        # -------------------------------
        if selected_song:
            idx = df[df[title_col] == selected_song].index[0]
            distances, indices = model.kneighbors([X_scaled[idx]])

            st.subheader("🎶 Rekomendasi Lagu Serupa:")
            recs = df.iloc[indices[0][1:]][[title_col] + FEATURES]
            st.dataframe(recs.reset_index(drop=True))
else:
    st.info("Ketik nama lagu di atas untuk mencari rekomendasi 🎵")

# -------------------------------
# 6️⃣ Footer
# -------------------------------
st.markdown("---")
st.caption("Dibuat oleh **Roy Simbolon** — Sistem Rekomendasi Musik berbasis KNN 🎧")
