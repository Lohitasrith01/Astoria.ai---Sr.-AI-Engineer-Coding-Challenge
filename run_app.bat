@echo off
echo 🚀 Starting AI Candidate Screener...
echo.

REM Activate virtual environment
echo 📦 Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if activation worked
if "%VIRTUAL_ENV%"=="" (
    echo ❌ Failed to activate virtual environment
    echo Please run: python -m venv venv
    echo Then run: venv\Scripts\activate
    pause
    exit /b 1
)

echo ✅ Virtual environment activated: %VIRTUAL_ENV%
echo.

REM Test imports
echo 🔍 Testing imports...
python -c "from src.resume_parser import load_resume_text; print('✅ Resume parser ready')" 2>nul
if errorlevel 1 (
    echo ❌ Import error detected. Installing requirements...
    pip install -r requirements.txt
)

echo.
echo 🌐 Starting Streamlit app...
echo 📱 App will open in your browser at: http://localhost:8501
echo.
echo 💡 Tips:
echo    - Use 'Load Sample Data' to test with demo candidates
echo    - Upload your own JSON/CSV files with candidate data
echo    - Try the 'Single Candidate Test' tab for quick demos
echo.

streamlit run app.py

echo.
echo 👋 App stopped. Press any key to exit...
pause >nul 