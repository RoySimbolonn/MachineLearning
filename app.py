import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="🎵 Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #1DB954;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1ed760;
        border: none;
    }
    h1 {
        color: #1DB954;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    """Load dataset dan siapkan fitur"""
    try:
        df = pd.read_csv("song_cleaned .csv")

        # cari kolom nama lagu
        possible_title_cols = ['song_name', 'title', 'name', 'track_name']
        title_col = next((c for c in possible_title_cols if c in df.columns), None)
        if title_col is None:
            st.error("❌ Kolom nama lagu tidak ditemukan!")
            return None, None, None

        if title_col != 'song_name':
            df = df.rename(columns={title_col: 'song_name'})

        FEATURES = [
            'danceability', 'energy', 'audio_valence', 'acousticness',
            'tempo', 'instrumentalness', 'speechiness', 'loudness'
        ]
        missing_feats = [f for f in FEATURES if f not in df.columns]
        if missing_feats:
            st.error(f"❌ Kolom tidak ditemukan: {missing_feats}")
            return None, None, None

        df_features = df[FEATURES].copy()

        scaler = MinMaxScaler()
        features_to_normalize = []
        if df_features['tempo'].max() > 1.5:
            features_to_normalize.append('tempo')
        if df_features['loudness'].min() < 0:
            features_to_normalize.append('loudness')

        if features_to_normalize:
            df_features[features_to_normalize] = scaler.fit_transform(df_features[features_to_normalize])

        metadata_cols = ['song_name']
        if 'artist_name' in df.columns:
            metadata_cols.append('artist_name')
        elif 'artist' in df.columns:
            df = df.rename(columns={'artist': 'artist_name'})
            metadata_cols.append('artist_name')

        df_metadata = df[metadata_cols].copy()
        return df_metadata, df_features, FEATURES

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None, None

df_metadata, df_features, FEATURES = load_data()
if df_metadata is None or df_features is None:
    st.stop()

# ==========================================
# MODEL
# ==========================================

@st.cache_resource
def train_model(features):
    model = NearestNeighbors(n_neighbors=30, metric='euclidean', algorithm='auto', n_jobs=-1)
    model.fit(features)
    return model

knn_model = train_model(df_features)

# ==========================================
# FUNCTIONS
# ==========================================

def calculate_diversity_score(candidate_features, selected_features_list):
    if len(selected_features_list) == 0:
        return 0
    distances = [np.linalg.norm(candidate_features - f) for f in selected_features_list]
    return np.mean(distances)

def get_recommendations(song_name, n_recommendations=10, diversity_weight=0.3):
    song_indices = df_metadata[df_metadata['song_name'].str.lower() == song_name.lower()].index
    if len(song_indices) == 0:
        return None, None
    song_idx = song_indices[0]
    input_song_name = df_metadata.iloc[song_idx]['song_name']
    song_features = df_features.iloc[song_idx].values.reshape(1, -1)
    pool_size = min(n_recommendations * 3, len(df_features) - 1)
    distances, indices = knn_model.kneighbors(song_features, n_neighbors=pool_size + 1)

    candidates, seen_songs = [], {input_song_name.lower()}
    for i in range(1, len(indices[0])):
        idx = indices[0][i]
        candidate_name = df_metadata.iloc[idx]['song_name']
        if candidate_name.lower() in seen_songs:
            continue
        seen_songs.add(candidate_name.lower())
        dist = distances[0][i]
        similarity = 1 / (1 + dist)
        candidates.append({
            'idx': idx,
            'song_name': candidate_name,
            'similarity': similarity,
            'features': df_features.iloc[idx].values
        })

    selected, selected_features = [], []
    for _ in range(n_recommendations):
        if len(candidates) == 0:
            break
        best_score, best_candidate = -1, None
        for c in candidates:
            sim_score = c['similarity']
            div_score = calculate_diversity_score(c['features'], selected_features)
            if len(selected_features) > 0:
                div_score = min(div_score / 0.5, 1.0)
            final_score = (1 - diversity_weight) * sim_score + diversity_weight * div_score
            if final_score > best_score:
                best_score, best_candidate = final_score, c
        if best_candidate:
            best_candidate['score'] = best_score
            selected.append(best_candidate)
            selected_features.append(best_candidate['features'])
            candidates.remove(best_candidate)

    recs = []
    for i, rec in enumerate(selected, 1):
        data = {
            'Rank': i,
            'Song': rec['song_name'],
            'Similarity': f"{rec['similarity']*100:.1f}%"
        }
        if 'artist_name' in df_metadata.columns:
            data['Artist'] = df_metadata.iloc[rec['idx']]['artist_name']
        recs.append(data)
    return pd.DataFrame(recs), df_features.iloc[song_idx].to_dict()

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.title("🎵 Settings")
    diversity_weight = st.slider("🎨 Diversity Weight", 0.0, 0.5, 0.3, 0.05)
    n_recommendations = st.slider("📊 Number of Recommendations", 5, 20, 10, 1)
    st.markdown("---")
    st.metric("Total Songs", f"{len(df_metadata):,}")
    st.metric("Algorithm", "Hybrid KNN")

# ==========================================
# MAIN
# ==========================================

st.title("🎵 Music Recommendation System")
st.markdown("### Find your next favorite song")

col1, col2 = st.columns([3, 1])
with col1:
    song_input = st.text_input(
        "🔍 Search song name:",
        placeholder="Type to search..."
    )

if song_input:
        matches = [s for s in df_metadata['song_name'].tolist() if song_input.lower() in s.lower()]
        if len(matches) > 0:
            st.markdown("**Suggestions:**")
            for i, s in enumerate(matches[:5]):  # tampilkan 5 teratas
                st.markdown(f"- 🎵 {s}")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_button = st.button("🎯 Get Recommendations", use_container_width=True)

if search_button and song_input:
    with st.spinner("🎧 Generating recommendations..."):
        recommendations, _ = get_recommendations(
            song_input, n_recommendations, diversity_weight
        )

    if recommendations is not None:
        st.success(f"✅ Found {len(recommendations)} recommendations for **{song_input}**")
        st.dataframe(recommendations, use_container_width=True)
    else:
        st.error(f"❌ Song '{song_input}' not found in database.")
elif search_button and not song_input:
    st.warning("⚠️ Please select a song first!")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🎵 Music Recommendation System — Developed by <strong>Roy Simbolon</strong></p>
    </div>
""", unsafe_allow_html=True)
