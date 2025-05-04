# 1. Update pip (optional, in case it's outdated)
Write-Host "🛠️ Updating pip..."
python -m pip install --upgrade pip

# 2. Create a virtual environment
Write-Host "🐍 Creating virtual environment..."
python -m venv venv

# 3. Activate the virtual environment
Write-Host "🔑 Activating virtual environment..."
.\venv\Scripts\Activate.ps1

# 4. Install required libraries from requirements.txt
Write-Host "📦 Installing required libraries from requirements.txt..."
pip install -r requirements.txt

# 5. Run the Streamlit app
Write-Host "🚀 Running Streamlit app..."
streamlit run app.py
