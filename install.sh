#!/usr/bin/env bash
# Memorial Bot — One-click setup for Mac/Linux
# Run this once after copying the memorial_bot folder to a new machine.

set -e

echo "=== Memorial Bot Setup ==="
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python 3 is required. Download from https://python.org"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python $PYTHON_VERSION found."

# Install Ollama if not present
if ! command -v ollama &>/dev/null; then
    echo ""
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama already installed: $(ollama --version)"
fi

# Install Python dependencies (serving + RAG only, no training deps)
echo ""
echo "Installing Python dependencies..."
python3 -m pip install --quiet gradio chromadb sentence-transformers ollama pyyaml

# Register model with Ollama
MODEL_FILE="serving/Modelfile"
if [ -f "$MODEL_FILE" ]; then
    MODEL_NAME=$(grep -m1 "^# ollama_name:" "$MODEL_FILE" | cut -d' ' -f3 || echo "memorial-bot")
    echo ""
    echo "Registering model with Ollama..."
    ollama create memorial-bot -f "$MODEL_FILE"
    echo "Model registered as 'memorial-bot'."
else
    echo ""
    echo "WARNING: serving/Modelfile not found."
    echo "  Make sure you copied the full deployment package."
fi

echo ""
echo "=== Setup complete! ==="
echo "Run './start.sh' to start the chatbot."
