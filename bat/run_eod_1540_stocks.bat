@echo off
set ROOT=C:\Users\Saarit\OneDrive\Desktop\Trading\algosm1\algo_trading
set LOG=%ROOT%\logs\eod_daily_weekly_1540_stocks.log

call C:\Users\Saarit\anaconda3\Scripts\activate.bat fin
cd /d %ROOT%

echo ================================ >> %LOG%
echo [%DATE% %TIME%] START EOD 1540 >> %LOG%

set PYTHONPATH=%ROOT%;%PYTHONPATH%
python scripts\schedulers\stocks_eod_daily_weekly_scheduler_for_daily_1540_update.py >> %LOG% 2>&1

echo [%DATE% %TIME%] END EOD 1540 >> %LOG%
