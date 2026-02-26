@echo off
set ROOT=C:\Users\Saarit\OneDrive\Desktop\Trading\algosm1\algo_trading
set LOG=%ROOT%\logs\eqidv1_eod_1540.log

call C:\Users\Saarit\anaconda3\Scripts\activate.bat fin
cd /d %ROOT%

echo ================================ >> %LOG%
echo [%DATE% %TIME%] START eqidv1 EOD 1540 >> %LOG%

set PYTHONPATH=%ROOT%;%PYTHONPATH%
python scripts\schedulers\eqidv1_eod_scheduler_for_1540_update.py >> %LOG% 2>&1

echo [%DATE% %TIME%] END eqidv1 EOD 1540 >> %LOG%
