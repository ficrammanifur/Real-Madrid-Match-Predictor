# Real Madrid Match Predictor

Aplikasi web untuk memprediksi hasil pertandingan Real Madrid menggunakan machine learning dan algoritma prediksi canggih dengan integrasi API football-data.org.

## 🏆 Fitur

- **Prediksi ML Akurat**: Model XGBoost dengan 30+ fitur termasuk Elo rating, xG, dan form tim
- **Data Real-time**: Integrasi dengan football-data.org API untuk skor dan pencetak gol
- **Interface Modern**: Design responsif dengan tema Real Madrid
- **Multi-kompetisi**: La Liga, Champions League, Copa del Rey
- **Analisis Mendalam**: Feature importance, confidence intervals, dan evaluasi model

## 📁 Struktur Folder

\`\`\`
real-madrid-predictor/
├── data/                            # 📊 FOLDER DATA
│   ├── raw/                         # Data mentah
│   │   ├── matches.csv              # Data pertandingan historis
│   │   ├── players.json             # Data pemain dan injury
│   │   └── team_stats.json          # Statistik tim (xG, possession, dll)
│   ├── processed/                   # Data yang sudah diproses
│   │   ├── features.csv             # Feature engineering results
│   │   └── model_data.csv           # Data siap untuk training
│   └── external/                    # Data dari API eksternal
│       ├── football_data_cache.json # Cache dari football-data.org
│       └── fixtures.json            # Jadwal pertandingan
├── models/                          # 🤖 MODEL ML
│   ├── trained_models/              # Model yang sudah dilatih
│   │   ├── xgboost_model.pkl        # Model XGBoost utama
│   │   ├── elo_ratings.json         # Rating Elo terkini
│   │   └── feature_scaler.pkl       # Scaler untuk normalisasi
│   └── evaluation/                  # Hasil evaluasi model
│       ├── model_metrics.json       # Akurasi, precision, recall
│       └── feature_importance.png   # Visualisasi feature importance
├── public/                          # 🌐 FRONTEND
│   ├── assets/
│   │   └── logo/
│   │       └── real-madrid-logo.png # Logo Real Madrid
│   ├── index.html                   # Frontend utama
│   ├── style.css                    # Styling
│   └── main.js                      # JavaScript logic
├── scripts/                         # 🔧 SCRIPT ML & DATA
│   ├── data_collector.py            # Pengumpulan data dari API
│   ├── feature_engineering.py       # Feature engineering
│   ├── train_model.py               # Training model ML
│   ├── model_evaluation.py          # Evaluasi dan validasi
│   ├── advanced_features.py         # Fitur lanjutan (Elo, xG)
│   └── ucl_simulation.py            # Simulasi Monte Carlo UCL
├── config/                          # ⚙️ KONFIGURASI
│   ├── api_keys.json                # API keys (JANGAN COMMIT!)
│   └── model_config.json            # Konfigurasi model
├── app.py                           # Backend Flask
├── requirements.txt                 # Dependencies Python
└── README.md                        # Dokumentasi
\`\`\`

## 🚀 Cara Menjalankan

### 1. Persiapan Environment

\`\`\`bash
# Clone atau download project
cd real-madrid-predictor

# Buat folder yang diperlukan
mkdir -p data/{raw,processed,external}
mkdir -p models/{trained_models,evaluation}
mkdir -p config

# Install dependencies Python
pip install -r requirements.txt
\`\`\`

### 2. Setup API Key

Buat file `config/api_keys.json`:
\`\`\`json
{
    "football_data_org": "YOUR_API_KEY_HERE"
}
\`\`\`

**Dapatkan API key gratis di**: https://www.football-data.org/client/register

### 3. Persiapan Data

\`\`\`bash
# Jalankan data collector untuk mengumpulkan data
python scripts/data_collector.py

# Generate features untuk ML
python scripts/feature_engineering.py

# Train model (opsional, sudah ada pre-trained model)
python scripts/train_model.py
\`\`\`

### 4. Tambahkan Logo

Letakkan logo Real Madrid di:
\`\`\`
public/assets/logo/real-madrid-logo.png
\`\`\`

### 5. Jalankan Server

\`\`\`bash
python app.py
\`\`\`

### 6. Buka Browser

Akses aplikasi di: **http://127.0.0.1:5000**

## 📊 Format Data

### matches.csv (Data Historis)
\`\`\`csv
date,home_team,away_team,home_score,away_score,competition,venue,home_xg,away_xg,home_possession,away_possession
2024-01-15,Real Madrid,Barcelona,2,1,La Liga,Home,2.3,1.8,58,42
\`\`\`

### players.json (Data Pemain)
\`\`\`json
{
    "Real Madrid": {
        "injuries": ["Vinicius Jr", "Courtois"],
        "suspensions": [],
        "key_players": ["Bellingham", "Mbappe", "Modric"]
    }
}
\`\`\`

### team_stats.json (Statistik Tim)
\`\`\`json
{
    "Real Madrid": {
        "recent_form": [1, 1, 0, 1, 1],
        "avg_xg_for": 2.1,
        "avg_xg_against": 0.9,
        "home_advantage": 0.15
    }
}
\`\`\`

## 🎯 Cara Penggunaan

### Mode Prediksi
1. **Input Data Pertandingan**:
   - Pilih tim lawan dari dropdown
   - Pilih kompetisi (La Liga, Champions League, Copa del Rey)
   - Pilih venue (Home/Away)
   - Atur parameter lanjutan (opsional)

2. **Dapatkan Prediksi**:
   - Klik "Prediksi Hasil"
   - Lihat probabilitas Menang/Seri/Kalah
   - Cek confidence interval dan feature importance

### Mode Data Real
1. **Hasil Terbaru**: Lihat skor dan pencetak gol pertandingan terakhir
2. **Jadwal**: Cek fixture mendatang Real Madrid
3. **Refresh Data**: Update data terbaru dari API

## 🧠 Algoritma ML

### Features (30+ variabel):
- **Elo Ratings**: Dynamic rating dengan momentum
- **Form Metrics**: Performa 5 pertandingan terakhir
- **xG Analytics**: Expected Goals untuk dan melawan
- **Venue Factors**: Home advantage, travel distance
- **Player Impact**: Injury/suspension key players
- **Tactical Context**: Head-to-head history, competition type

### Model Pipeline:
1. **Data Collection**: API + historical data
2. **Feature Engineering**: 30+ engineered features
3. **Model Training**: XGBoost dengan hyperparameter tuning
4. **Validation**: Time-based split, cross-validation
5. **Calibration**: Probability calibration untuk confidence

## 🛠️ Teknologi

- **Backend**: Python Flask, XGBoost, scikit-learn
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Data**: football-data.org API, pandas, numpy
- **ML**: XGBoost, feature engineering, model evaluation
- **Styling**: Custom CSS dengan tema Real Madrid

## 📱 Responsive Design

Aplikasi dioptimalkan untuk:
- Desktop (1200px+): Full dashboard dengan semua fitur
- Tablet (768px - 1199px): Layout adaptif
- Mobile (< 768px): Interface mobile-first

## 🔧 Kustomisasi

### Menambah Data Baru
\`\`\`bash
# Update data dari API
python scripts/data_collector.py --update

# Re-train model dengan data baru
python scripts/train_model.py --retrain
\`\`\`

### Mengubah Model
Edit konfigurasi di `config/model_config.json`:
\`\`\`json
{
    "model_type": "xgboost",
    "hyperparameters": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1
    }
}
\`\`\`

### API Endpoints
- `POST /predict`: Prediksi hasil pertandingan
- `GET /recent_matches`: Hasil pertandingan terbaru
- `GET /fixtures`: Jadwal pertandingan
- `GET /model_info`: Informasi model dan akurasi

## 📈 Evaluasi Model

Model dievaluasi menggunakan:
- **Accuracy**: ~73% pada test set
- **Log Loss**: Optimized untuk probability calibration
- **Brier Score**: Reliability of probability predictions
- **Feature Importance**: Top features yang mempengaruhi prediksi

## 🏅 Hala Madrid!

Dibuat dengan ❤️ untuk Madridistas di seluruh dunia menggunakan teknologi ML terdepan.

---

**Catatan**: Pastikan untuk tidak commit file `config/api_keys.json` ke repository publik!
