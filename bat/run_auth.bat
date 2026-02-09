@echo off
set ROOT=C:\Users\Saarit\OneDrive\Desktop\Trading\algosm1\algo_trading
set LOG=%ROOT%\logs\auth.log

call C:\Users\Saarit\anaconda3\Scripts\activate.bat fin
cd /d %ROOT%

echo ================================ >> %LOG%
echo [%DATE% %TIME%] START auth >> %LOG%

python algosm1_authentication.py >> %LOG% 2>&1

echo [%DATE% %TIME%] END auth >> %LOG%
