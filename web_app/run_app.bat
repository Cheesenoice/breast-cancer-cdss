@echo off
title Multimodal AI CDSS - TCGA-BRCA Web Application
echo =========================================================================
echo       KHOI DONG HE THONG HO TRO QUYET DINH LAM SANG (CDSS WEB DEMO)
echo =========================================================================
echo.
echo Dang khoi chay may chu Streamlit tren trinh duyet...
echo.

cd /d "%~dp0"
"C:\Users\huynh\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run app.py --server.port 8501 --server.headless false

pause
