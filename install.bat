@echo off
REM Memorial Bot — One-click setup for Windows
REM Run this once after copying the memorial_bot folder to a new machine.

echo === Memorial Bot Setup ===
echo.

REM Check Python
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo ERROR: Python 3 is required. Download from https://python.org
    pause
    exit /b 1
)

echo Python found.

REM Check if Ollama is installed
ollama --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo.
    echo Ollama not found. Please install it from https://ollama.com/download
    echo After installing Ollama, re-run this script.
    pause
    exit /b 1
) ELSE (
    echo Ollama found.
)

REM Install Python dependencies
echo.
echo Installing Python dependencies...
python -m pip install --quiet gradio chromadb sentence-transformers ollama pyyaml

REM Register model with Ollama
IF EXIST "serving\Modelfile" (
    echo.
    echo Registering model with Ollama...
    ollama create memorial-bot -f serving\Modelfile
    echo Model registered as 'memorial-bot'.
) ELSE (
    echo.
    echo WARNING: serving\Modelfile not found.
    echo   Make sure you copied the full deployment package.
)

echo.
echo === Setup complete! ===
echo Run 'start.bat' to start the chatbot.
pause
