@echo off
set ROOT=C:\Users\Saarit\OneDrive\Desktop\Trading\algosm1\algo_trading
set LOG=%ROOT%\logs\live_15m_updater_stocks.log

call C:\Users\Saarit\anaconda3\Scripts\activate.bat fin
cd /d %ROOT%

echo ================================ >> %LOG%
echo [%DATE% %TIME%] START 15m updater >> %LOG%

set PYTHONPATH=%ROOT%;%PYTHONPATH%
python scripts\schedulers\stocks_eod_daily_weekly_scheduler_for_15mins_data.py >> %LOG% 2>&1

echo [%DATE% %TIME%] END 15m updater >> %LOG%
