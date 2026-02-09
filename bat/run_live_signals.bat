@echo off
set ROOT=C:\Users\Saarit\OneDrive\Desktop\Trading\algosm1\algo_trading
set LOG=%ROOT%\logs\live_signals_15m.log

call C:\Users\Saarit\anaconda3\Scripts\activate.bat fin
cd /d %ROOT%

echo ================================ >> %LOG%
echo [%DATE% %TIME%] START live signals >> %LOG%

python etf_live_trading_signal_15m_v7_parquet.py >> %LOG% 2>&1

echo [%DATE% %TIME%] END live signals >> %LOG%
