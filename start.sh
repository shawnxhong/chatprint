#!/usr/bin/env bash
# Memorial Bot — Start the chatbot
# Opens the web UI at http://localhost:7860

set -e

echo "=== Starting Memorial Bot ==="

# Start Ollama in background if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    OLLAMA_PID=$!
    sleep 2
    echo "Ollama started (PID $OLLAMA_PID)"
fi

# Start the Gradio app
echo "Starting chat interface at http://localhost:7860"
echo "Press Ctrl+C to stop."
echo ""
python3 serving/app.py --config config.yaml
