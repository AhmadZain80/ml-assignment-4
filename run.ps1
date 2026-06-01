# Interactive ML Dashboard - PowerShell Launch Script

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   📊 Interactive ML Classification Dashboard" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"

# Clear screen for clarity
Clear-Host

Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   📊 Interactive ML Classification Dashboard" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "🚀 Starting Streamlit application..." -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Application URL: http://localhost:8501" -ForegroundColor Cyan
Write-Host "📝 API: http://localhost:8501/_stcore/health" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Run Streamlit app
streamlit run app.py
