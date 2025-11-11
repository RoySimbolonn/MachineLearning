# 🎵 Penerapan Algoritma K-Nearest Neighbors (KNN) untuk Sistem Rekomendasi Lagu Berbasis Content-Based Recommendation

## 📘 Deskripsi Proyek
Proyek ini merupakan implementasi sistem rekomendasi lagu berbasis **content-based filtering** menggunakan algoritma **K-Nearest Neighbors (KNN)** yang dikombinasikan dengan pendekatan **Maximal Marginal Relevance (MMR)** untuk meningkatkan *diversity* hasil rekomendasi.  
Sistem mampu memberikan rekomendasi lagu yang mirip dengan lagu input berdasarkan kemiripan karakteristik fitur audio seperti **acousticness, energy, danceability, loudness**, dan sebagainya.

Proyek ini merupakan bagian dari **Ujian Tengah Semester Mata Kuliah Machine Learning** – Semester Ganjil Tahun Akademik 2025/2026 di Universitas Mikroskil.

---

## 👥 Anggota Tim

Roy Jannes Simbolon (22113506)
Saumel Natalino Sitorus (221111771)
Dela Amelia Sitorus (221112198)

---

## 🧩 Kompleksitas Masalah
Masalah utama adalah menentukan kemiripan antar lagu berdasarkan fitur numerik yang diperoleh dari metadata audio Spotify.  
Kompleksitasnya meliputi:
- **Tingginya dimensi data** (8 fitur numerik utama)
- **Perbedaan skala antar fitur**, memerlukan normalisasi
- **Adanya duplikat (3.909 baris)** dan **outlier** pada data sehingga dataset menjadi 14,926
- **Tidak adanya label ground truth**, sehingga evaluasi berbasis kemiripan

---

## 🗃 Dataset
- **Sumber:** Kaggle (Spotify Dataset – Audio Features)  
- **Jumlah data:** 18.835 baris × 15 kolom  
- **Fitur utama:** `['acousticness', 'danceability', 'energy', 'liveness', 'loudness', 'speechiness', 'tempo', 'audio_valence']`  
- **Jenis data:** Real-world dataset (fitur diekstraksi dari Spotify API)  

---

## 🔍 Exploratory Data Analysis (EDA)
Beberapa tahapan analisis awal dilakukan untuk memahami data:
- Distribusi fitur audio divisualisasikan menggunakan histogram
- Korelasi antar fitur divisualisasikan dalam heatmap
- Ditemukan korelasi tinggi antara beberapa fitur seperti *energy* dan *loudness*
- Outlier ditemukan pada fitur *speechiness* (≈ 12.2%)

---

## 🧹 Preprocessing Data
1. **Menghapus duplikat:** 3.909 baris duplikat dihapus dengan `drop_duplicates()`
2. **Memeriksa missing value:** Tidak ditemukan nilai kosong
3. **Normalisasi:** Menggunakan `MinMaxScaler()` agar seluruh fitur memiliki rentang [0, 1]
4. **Penanganan outlier:** Metode IQR digunakan untuk deteksi dan pembersihan outlier

---

## 🤖 Model & Algoritma
### Algoritma Utama
- **K-Nearest Neighbors (KNN)**  
- **Distance Metric:** Euclidean Distance  
- **Parameter:**
  - K = 30  
  - Output rekomendasi: 10 lagu teratas  
  - α = 0.3 (diversity weight)

### Hybrid KNN + Maximal Marginal Relevance (MMR)
Integrasi similarity dan diversity:
```
final_score = (1 - α) × similarity + α × diversity
```

Tahapan:
1. KNN memilih 30 lagu terdekat  
2. MMR memilih 10 terbaik dengan keseimbangan similarity & diversity  

---

## 📊 Evaluasi Model
### Metrik Evaluasi
- **Similarity Score:** `1 / (1 + distance)`
- **Precision@10**, **Recall@10**, **F1-score**
- **NDCG@10** – kualitas ranking rekomendasi
- **MAP@10** – rata-rata presisi
- **Intra-List Diversity** – keragaman antar hasil rekomendasi

### Hasil (contoh)
| Metrik | Nilai |
|--------|--------|
| Precision@10 | 0.86 |
| Recall@10 | 0.80 |
| F1-score | 0.83 |
| NDCG@10 | 0.87 |
| MAP@10 | 0.84 |
| Coverage | 91% |

---

## ⚙️ Teknologi & Tools
- **Bahasa:** Python 3.10  
- **Library Utama:**
  - `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
  - `streamlit`, `plotly`
- **Lingkungan Pengembangan:** Google Colab  
- **Framework Deployment:** Streamlit  

---

## 🚀 Deployment & Demo
- **Repository GitHub:** [https://github.com/RoySimbolonn/MachineLearning](https://github.com/RoySimbolonn/MachineLearning)  
- **Aplikasi Streamlit:** [https://machinelearning-dztjkyfwvafauh8ryfdfdd.streamlit.app/](https://machinelearning-dztjkyfwvafauh8ryfdfdd.streamlit.app/)  
- **Video Penjelasan:** [Tonton di sini](https://mikroskilacid-my.sharepoint.com/:v:/g/personal/221113506_students_mikroskil_ac_id/ERzDHwlP1SZMjN-viBisNfoB-ppMORtrNbjqg3u1Iw5HHg?e=OSl2N0)

---

## 💻 Cara Menjalankan Proyek

### 1️⃣ Clone Repository
```bash
git clone https://github.com/RoySimbolonn/MachineLearning.git
cd MachineLearning
```

### 2️⃣ Install Dependencies
Pastikan sudah terpasang **Python ≥3.10** dan `pip`.  
Kemudian jalankan:
```bash
pip install -r requirements.txt
```

### 3️⃣ Jalankan Notebook Utama
Gunakan Jupyter Notebook atau VSCode:
```bash
jupyter notebook CollabLink.ipynb
```

Atau langsung jalankan aplikasi Streamlit (jika tersedia):
```bash
streamlit run streamlit_app.py
```

---

## 📂 Struktur Repository
```
├── data/                     ← Dataset dan hasil preprocessing
├── models/                   ← Model dan checkpoint KNN
├── CollabLink.ipynb          ← Notebook utama
├── streamlit_app.py          ← Aplikasi Streamlit (deployment)
├── requirements.txt          ← Daftar library
└── README.md                 ← Dokumentasi proyek
```

---

## 🧮 Pembagian Tugas
| Anggota | Tugas Utama |
|----------|--------------|
| **Samuel Natalino Sitorus** |Mencari Dataset, Data preprocessing, analisis hasil |
| **Roy Jannes Simbolon** | Implementasi & hybird model, deployment hosting,  video penjelasan, pembuatan repository GitHub |
| **Dela Amelia Sitorus** | Visualisasi hasil, penyusunan laporan, mengatur struktur notebook |

---

## 🪪 Lisensi
Proyek ini dibuat untuk keperluan akademik dan pembelajaran.  
Lisensi: **MIT License** – Bebas digunakan dengan mencantumkan atribusi.

---

> “Sistem rekomendasi ini membuktikan bahwa pendekatan sederhana seperti KNN, bila dikombinasikan dengan metode diversifikasi yang tepat, dapat menghasilkan rekomendasi yang relevan dan bervariasi secara efisien.”
