@echo off
REM Interactive ML Dashboard - Launch Script

echo.
echo ════════════════════════════════════════════════════════════
echo   📊 Interactive ML Classification Dashboard
echo ════════════════════════════════════════════════════════════
echo.

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Run Streamlit app
echo.
echo Starting Streamlit application...
echo.
echo 🌐 Application will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
streamlit run app.py

pause
