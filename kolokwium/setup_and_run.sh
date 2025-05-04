#!/bin/bash

# 1. Check if Python and pip are installed
echo "🛠️ Checking if Python and pip are installed..."

# Check for Python3
if ! command -v python3 &>/dev/null; then
    echo "Python3 is not installed. Please install Python before running this script."
    exit 1
fi

# Check for pip3
if ! command -v pip3 &>/dev/null; then
    echo "pip3 is not installed. Please install pip before running this script."
    exit 1
fi

# 2. Update pip
echo "📦 Updating pip..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # On macOS, use the --break-system-packages flag
    python3 -m pip install --upgrade pip --break-system-packages
else
    python3 -m pip install --upgrade pip
fi

# 3. Create a virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv venv

# 4. Activate the virtual environment based on OS
echo "🔑 Activating virtual environment..."

# Check for Windows (msys is used by Git Bash or PowerShell)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Windows Git Bash or Cygwin environment
    source venv/Scripts/activate
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux environment
    source venv/bin/activate
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS environment
    source venv/bin/activate
else
    echo "Unsupported operating system. This script only works on Windows, Linux, and macOS."
    exit 1
fi

# 5. Install required libraries
echo "📦 Installing required libraries from requirements.txt..."
pip install --upgrade -r requirements.txt --break-system-packages

# 6. Run the Streamlit app
echo "🚀 Running Streamlit app..."
streamlit run app.py
