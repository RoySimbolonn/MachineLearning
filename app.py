import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="🎵 Music Recommendation System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
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
        # Load data
        df = pd.read_csv("song_cleaned .csv")
        
        # Deteksi kolom nama lagu
        possible_title_cols = ['song_name', 'title', 'name', 'track_name']
        title_col = next((c for c in possible_title_cols if c in df.columns), None)
        
        if title_col is None:
            st.error("❌ Kolom nama lagu tidak ditemukan!")
            return None, None, None
        
        # Rename to song_name for consistency
        if title_col != 'song_name':
            df = df.rename(columns={title_col: 'song_name'})
        
        # Fitur yang digunakan
        FEATURES = [
            'danceability', 'energy', 'audio_valence', 'acousticness',
            'tempo', 'instrumentalness', 'speechiness', 'loudness'
        ]
        
        # Cek missing features
        missing_feats = [f for f in FEATURES if f not in df.columns]
        if missing_feats:
            st.error(f"❌ Kolom tidak ditemukan: {missing_feats}")
            return None, None, None
        
        # Extract features
        df_features = df[FEATURES].copy()
        
        # Normalisasi (Min-Max untuk tempo dan loudness)
        scaler = MinMaxScaler()
        features_to_normalize = []
        
        # Cek range untuk tempo
        if df_features['tempo'].max() > 1.5:
            features_to_normalize.append('tempo')
        
        # Cek range untuk loudness
        if df_features['loudness'].min() < 0:
            features_to_normalize.append('loudness')
        
        if features_to_normalize:
            df_features[features_to_normalize] = scaler.fit_transform(
                df_features[features_to_normalize]
            )
        
        # Extract metadata
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

# Load data
df_metadata, df_features, FEATURES = load_data()

if df_metadata is None or df_features is None:
    st.stop()

# ==========================================
# TRAIN KNN MODEL
# ==========================================

@st.cache_resource
def train_model(features):
    """Train KNN model"""
    model = NearestNeighbors(
        n_neighbors=30,
        metric='euclidean',
        algorithm='auto',
        n_jobs=-1
    )
    model.fit(features)
    return model

knn_model = train_model(df_features)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def calculate_diversity_score(candidate_features, selected_features_list):
    """Calculate diversity score"""
    if len(selected_features_list) == 0:
        return 0
    
    distances = []
    for selected_feat in selected_features_list:
        dist = np.linalg.norm(candidate_features - selected_feat)
        distances.append(dist)
    
    return np.mean(distances)


def get_recommendations(song_name, n_recommendations=10, diversity_weight=0.3):
    """Get hybrid recommendations with diversity boost"""
    
    # Find song index (case-insensitive)
    song_indices = df_metadata[df_metadata['song_name'].str.lower() == song_name.lower()].index
    
    if len(song_indices) == 0:
        return None, None
    
    song_idx = song_indices[0]
    input_song_name = df_metadata.iloc[song_idx]['song_name']
    
    # Get features
    song_features = df_features.iloc[song_idx].values.reshape(1, -1)
    
    # Get neighbors
    pool_size = min(n_recommendations * 3, len(df_features) - 1)
    distances, indices = knn_model.kneighbors(song_features, n_neighbors=pool_size + 1)
    
    # Build candidate pool (remove duplicates)
    candidates = []
    seen_songs = set()
    seen_songs.add(input_song_name.lower())
    
    for i in range(1, len(indices[0])):
        idx = indices[0][i]
        candidate_song_name = df_metadata.iloc[idx]['song_name']
        
        if candidate_song_name.lower() in seen_songs:
            continue
        
        seen_songs.add(candidate_song_name.lower())
        
        dist = distances[0][i]
        similarity = 1 / (1 + dist)
        
        candidates.append({
            'idx': idx,
            'song_name': candidate_song_name,
            'similarity': similarity,
            'distance': dist,
            'features': df_features.iloc[idx].values
        })
    
    # MMR Selection
    selected = []
    selected_features = []
    remaining = candidates.copy()
    
    for _ in range(n_recommendations):
        if len(remaining) == 0:
            break
        
        best_score = -1
        best_candidate = None
        best_idx = -1
        
        for i, candidate in enumerate(remaining):
            sim_score = candidate['similarity']
            
            div_score = calculate_diversity_score(
                candidate['features'], 
                selected_features
            )
            
            if len(selected_features) > 0:
                div_score = min(div_score / 0.5, 1.0)
            else:
                div_score = 0
            
            final_score = (1 - diversity_weight) * sim_score + diversity_weight * div_score
            
            if final_score > best_score:
                best_score = final_score
                best_candidate = candidate
                best_idx = i
        
        if best_candidate:
            best_candidate['diversity_score'] = calculate_diversity_score(
                best_candidate['features'], 
                selected_features
            )
            best_candidate['final_score'] = best_score
            selected.append(best_candidate)
            selected_features.append(best_candidate['features'])
            remaining.pop(best_idx)
    
    # Format results
    recommendations = []
    for i, rec in enumerate(selected, 1):
        rec_data = {
            'Rank': i,
            'Song': rec['song_name'],
            'Similarity': f"{rec['similarity']*100:.1f}%",
            'Diversity': f"{rec['diversity_score']:.3f}",
            'Score': f"{rec['final_score']:.3f}"
        }
        
        # Add artist if available
        if 'artist_name' in df_metadata.columns:
            artist = df_metadata.iloc[rec['idx']]['artist_name']
            rec_data['Artist'] = artist
        
        recommendations.append(rec_data)
    
    # Get input song features
    input_features = df_features.iloc[song_idx].to_dict()
    
    return pd.DataFrame(recommendations), input_features


def create_feature_comparison_chart(input_features, rec_features_list, song_names):
    """Create radar chart comparing features"""
    
    feature_names = list(input_features.keys())
    
    fig = go.Figure()
    
    # Input song
    fig.add_trace(go.Scatterpolar(
        r=list(input_features.values()),
        theta=feature_names,
        fill='toself',
        name='Input Song',
        line=dict(color='#1DB954', width=3)
    ))
    
    # Top 3 recommendations
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i, (features, song_name, color) in enumerate(zip(rec_features_list[:3], song_names[:3], colors)):
        fig.add_trace(go.Scatterpolar(
            r=list(features.values()),
            theta=feature_names,
            fill='toself',
            name=f"{i+1}. {song_name[:30]}...",
            line=dict(color=color, width=2),
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title="Audio Features Comparison",
        height=500
    )
    
    return fig


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/musical-notes.png", width=80)
    st.title("🎵 Settings")
    
    st.markdown("---")
    
    # Diversity weight slider
    diversity_weight = st.slider(
        "🎨 Diversity Weight",
        min_value=0.0,
        max_value=0.5,
        value=0.3,
        step=0.05,
        help="Higher value = more diverse recommendations"
    )
    
    # Number of recommendations
    n_recommendations = st.slider(
        "📊 Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )
    
    st.markdown("---")
    
    # System info
    st.subheader("📈 System Info")
    st.metric("Total Songs", f"{len(df_metadata):,}")
    st.metric("Features Used", len(FEATURES))
    st.metric("Algorithm", "Hybrid KNN")
    
    st.markdown("---")
    
    # About
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Music Recommendation System**
        
        Built with:
        - K-Nearest Neighbors (KNN)
        - Maximal Marginal Relevance (MMR)
        - 8 Audio Features
        
        Developed by: Roy Simbolon
        """)

# ==========================================
# MAIN APP
# ==========================================

# Header
st.title("🎵 Music Recommendation System")
st.markdown("### Find your next favorite song based on audio similarity")

# Search section
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col1:
    # Song search with autocomplete
    song_input = st.selectbox(
        "🔍 Select or type a song name:",
        options=[''] + sorted(df_metadata['song_name'].tolist()),
        format_func=lambda x: "Type to search..." if x == '' else x
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_button = st.button("🎯 Get Recommendations", use_container_width=True)

# Get recommendations
if search_button and song_input:
    with st.spinner("🎵 Finding similar songs..."):
        recommendations, input_features = get_recommendations(
            song_input, 
            n_recommendations=n_recommendations,
            diversity_weight=diversity_weight
        )
    
    if recommendations is not None:
        st.success(f"✅ Found {len(recommendations)} recommendations for **{song_input}**")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_sim = float(recommendations['Similarity'].str.rstrip('%').astype(float).mean())
            st.metric("Avg Similarity", f"{avg_sim:.1f}%")
        with col2:
            avg_div = float(recommendations['Diversity'].astype(float).mean())
            st.metric("Avg Diversity", f"{avg_div:.3f}")
        with col3:
            unique = recommendations['Song'].nunique()
            st.metric("Unique Songs", f"{unique}/{len(recommendations)}")
        with col4:
            top_sim = float(recommendations.iloc[0]['Similarity'].rstrip('%'))
            st.metric("Top Match", f"{top_sim:.1f}%")
        
        st.markdown("---")
        
        # Results in tabs
        tab1, tab2, tab3 = st.tabs(["📋 Recommendations", "📊 Feature Analysis", "🎨 Visualization"])
        
        with tab1:
            st.subheader("🎵 Recommended Songs")
            
            # Display recommendations
            for idx, row in recommendations.iterrows():
                with st.container():
                    col1, col2, col3, col4 = st.columns([1, 4, 2, 2])
                    
                    with col1:
                        st.markdown(f"### {row['Rank']}")
                    
                    with col2:
                        st.markdown(f"**{row['Song']}**")
                        if 'Artist' in row:
                            st.caption(f"🎤 {row['Artist']}")
                    
                    with col3:
                        st.markdown(f"🎯 Similarity: **{row['Similarity']}**")
                    
                    with col4:
                        st.markdown(f"🎨 Diversity: **{row['Diversity']}**")
                    
                    st.markdown("---")
        
        with tab2:
            st.subheader("📊 Audio Features Analysis")
            
            # Feature comparison table
            st.markdown("#### Input Song Features:")
            feature_df = pd.DataFrame({
                'Feature': list(input_features.keys()),
                'Value': [f"{v:.3f}" for v in input_features.values()]
            })
            st.dataframe(feature_df, use_container_width=True)
            
            # Feature distribution
            st.markdown("#### Feature Distribution (Input vs Recommendations)")
            
            # Get features for top 3 recommendations
            rec_features_list = []
            for _, row in recommendations.head(3).iterrows():
                song_name = row['Song']
                idx = df_metadata[df_metadata['song_name'] == song_name].index[0]
                features = df_features.iloc[idx].to_dict()
                rec_features_list.append(features)
            
            # Create comparison dataframe
            comparison_data = {'Feature': list(input_features.keys())}
            comparison_data['Input'] = list(input_features.values())
            
            for i, features in enumerate(rec_features_list[:3]):
                comparison_data[f"Rec {i+1}"] = list(features.values())
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Bar chart
            fig = px.bar(
                comparison_df.melt(id_vars=['Feature'], var_name='Song', value_name='Value'),
                x='Feature',
                y='Value',
                color='Song',
                barmode='group',
                title='Feature Comparison',
                height=400
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("🎨 Feature Visualization")
            
            # Radar chart
            song_names = recommendations.head(3)['Song'].tolist()
            radar_chart = create_feature_comparison_chart(
                input_features,
                rec_features_list,
                song_names
            )
            st.plotly_chart(radar_chart, use_container_width=True)
            
            # Similarity distribution
            st.markdown("#### Similarity Score Distribution")
            sim_values = [float(x.rstrip('%')) for x in recommendations['Similarity']]
            fig = px.histogram(
                x=sim_values,
                nbins=10,
                title='Recommendation Similarity Distribution',
                labels={'x': 'Similarity (%)', 'y': 'Count'},
                color_discrete_sequence=['#1DB954']
            )
            st.plotly_chart(fig, use_container_width=True)
    
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
        <p>🎵 Music Recommendation System | Built with Streamlit & KNN</p>
        <p>Developed by <strong>Roy Simbolon</strong> — Powered by Hybrid K-Nearest Neighbors</p>
    </div>
""", unsafe_allow_html=True)