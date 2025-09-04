# 🚀 Setup Guide - Real Madrid Predictor

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **API Key** from [football-data.org](https://www.football-data.org/client/register)

## 🔧 Installation Steps

### 1. Clone/Download Project
\`\`\`bash
# If using git
git clone <repository-url>
cd real-madrid-predictor

# Or download and extract ZIP
\`\`\`

### 2. Create Virtual Environment
\`\`\`bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux  
source venv/bin/activate
\`\`\`

### 3. Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 4. Setup Project Structure
\`\`\`bash
python scripts/setup_project.py
\`\`\`

### 5. Configure API Keys
Edit `config/api_keys.json`:
\`\`\`json
{
  "football_data_org": "YOUR_ACTUAL_API_KEY_HERE"
}
\`\`\`

### 6. Add Real Madrid Logo
Place logo file in: `public/assets/logo/real-madrid-logo.png`

### 7. Collect Initial Data
\`\`\`bash
python scripts/data_collector.py
\`\`\`

### 8. Train Model (Optional)
\`\`\`bash
python scripts/train_model.py
\`\`\`

### 9. Run Application
\`\`\`bash
python3 app.py
\`\`\`

Open browser: `http://127.0.0.1:5000`

## 📁 File Locations

| Type | Location | Description |
|------|----------|-------------|
| **Match Data** | `data/raw/matches.csv` | Historical match results |
| **Player Data** | `data/raw/players.json` | Player stats and injuries |
| **API Cache** | `data/external/` | Cached API responses |
| **Models** | `models/trained_models/` | Trained ML models |
| **Logs** | `logs/` | Application logs |
| **Frontend** | `public/` | HTML, CSS, JS files |

## 🔑 API Keys Required

- **football-data.org**: Free tier (10 requests/minute)
- Get yours at: https://www.football-data.org/client/register

## 🚨 Important Notes

- Never commit `config/api_keys.json` to version control
- Model files are large - excluded from git by default
- API has rate limits - data collector handles this automatically