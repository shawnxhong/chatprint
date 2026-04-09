@echo off
REM Memorial Bot — Start the chatbot (Windows)
REM Opens the web UI at http://localhost:7860

echo === Starting Memorial Bot ===

REM Start Ollama server in background
start /B ollama serve
timeout /t 2 /nobreak >nul

REM Start the Gradio app
echo Starting chat interface at http://localhost:7860
echo Press Ctrl+C to stop.
echo.
python serving\app.py --config config.yaml
pause
