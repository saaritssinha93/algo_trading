@echo off
set ROOT=C:\Users\Saarit\OneDrive\Desktop\Trading\algosm1\algo_trading
set LOG=%ROOT%\logs\eqidv1_live_signals_15m.log

call C:\Users\Saarit\anaconda3\Scripts\activate.bat fin
cd /d %ROOT%

echo ================================ >> %LOG%
echo [%DATE% %TIME%] START eqidv1 live signals >> %LOG%

set PYTHONPATH=%ROOT%;%PYTHONPATH%
python scripts\live_trading\eqidv1_live_trading_signal_15m_v11_combined_parquet.py >> %LOG% 2>&1

echo [%DATE% %TIME%] END eqidv1 live signals >> %LOG%
