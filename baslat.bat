@echo off
REM Bu dosyaya çift tıklayarak uygulamayı başlatabilirsin.
REM Sanal ortamı otomatik aktif eder ve Streamlit'i çalıştırır.

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo HATA: .venv klasoru bulunamadi. Once kurulumu tamamlaman gerekiyor.
    echo Bkz. README.md dosyasindaki kurulum adimlari.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo Ollama'nin calistigini kontrol ediyorum...
where ollama >nul 2>nul
if errorlevel 1 (
    echo UYARI: 'ollama' komutu bulunamadi. Ollama'nin kurulu ve calisir durumda oldugundan emin ol.
)

echo.
echo Uygulama baslatiliyor, tarayicinda otomatik acilacak...
echo Kapatmak icin bu pencereyi kapatabilir ya da Ctrl+C basabilirsin.
echo.

python -m streamlit run app.py

pause
