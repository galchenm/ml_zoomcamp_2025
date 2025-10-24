# === Step 1: Set up paths ===
SRC_PATH="/mnt/c/Users/galch/OneDrive/Документы/GitHub/ml_zoomcamp_2025"
DST_PATH="$HOME/projects/ml_zoomcamp_2025"

echo "🔄 Copying project from OneDrive to WSL home..."
mkdir -p "$(dirname "$DST_PATH")"
cp -r "$SRC_PATH" "$DST_PATH"

cd "$DST_PATH/05-deployment/hw05" || { echo "❌ Folder not found"; exit 1; }

# === Step 2: Recreate venv ===
echo "🧹 Removing old virtual environment..."
rm -rf .venv

echo "�� Creating new virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# === Step 3: Install dependencies ===
echo "📦 Installing dependencies..."
uv pip install --upgrade pip
uv pip install fastapi uvicorn scikit-learn requests

# === Step 4: Test run ===
echo "🚀 Starting FastAPI app..."
uvicorn app:app --reload

