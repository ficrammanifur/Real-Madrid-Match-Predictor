import os
import json
from pathlib import Path

def create_project_structure():
    """Create the complete folder structure for Real Madrid Predictor"""
    
    # Define folder structure
    folders = [
        "data/raw",
        "data/processed", 
        "data/external",
        "models/trained_models",
        "models/evaluation",
        "public/assets/logo",
        "scripts",
        "config",
        "logs"
    ]
    
    # Create folders
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created folder: {folder}")
    
    # Create placeholder files
    placeholder_files = {
        "data/raw/.gitkeep": "",
        "data/processed/.gitkeep": "",
        "data/external/.gitkeep": "",
        "models/trained_models/.gitkeep": "",
        "models/evaluation/.gitkeep": "",
        "logs/.gitkeep": "",
        "public/assets/logo/README.md": "# Logo Folder\n\nTaruh logo Real Madrid di sini:\n- real-madrid-logo.png\n- real-madrid-logo.svg"
    }
    
    for file_path, content in placeholder_files.items():
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"✅ Created file: {file_path}")
    
    print("\n🎉 Project structure created successfully!")
    print("\n📋 Next steps:")
    print("1. Add your football-data.org API key to config/api_keys.json")
    print("2. Place Real Madrid logo in public/assets/logo/")
    print("3. Run: python scripts/data_collector.py")
    print("4. Run: python app.py")

if __name__ == "__main__":
    create_project_structure()