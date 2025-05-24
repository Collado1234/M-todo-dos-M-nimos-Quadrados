timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

if [ -d ".venv" ]; then
    echo "[$(timestamp)] Virtual environment already exists."
else
    echo "[$(timestamp)] Creating virtual environment..."
    python3 -m venv .venv
fi

echo "[$(timestamp)] Activating virtual environment..."

if [ -f ".venv/bin/activate" ]; then
    # For Unix/Linux/MacOS
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    # For Windows
    source .venv/Scripts/activate
else
    echo "[$(timestamp)] Error: Virtual environment activation script not found."
    exit 1
fi

echo "[$(timestamp)] Installing dependencies..."
python3 -m pip install -r requirements.txt
echo "[$(timestamp)] Running the script..."
python3 main.py
echo "[$(timestamp)] Script finished."
echo "[$(timestamp)] Deactivating virtual environment..."
deactivate
