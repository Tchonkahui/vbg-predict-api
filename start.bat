@echo off
cd C:\Users\Tchonkahui\Desktop\vbg_api
call venv\Scripts\activate
start http://localhost:8000/dashboard
python run.py
pause