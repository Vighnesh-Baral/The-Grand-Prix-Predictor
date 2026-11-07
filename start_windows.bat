@echo off
setlocal
start "The Grand Prix Predictor API" cmd /k "call .venv\Scripts\activate.bat && python run.py"
start "The Grand Prix Predictor Frontend" cmd /k "cd frontend && npm run dev"
