import os
import logging
import time
import threading
from datetime import datetime, timedelta, time as dt_time
import pandas as pd
import pytz
from kiteconnect import KiteConnect
import traceback
import glob
import signal
import sys
from logging.handlers import TimedRotatingFileHandler
from tqdm import tqdm
import json
from filelock import FileLock, Timeout
import schedule   # For time-based scheduling

# ================================
# Configuration
# ================================
india_tz = pytz.timezone('Asia/Kolkata')

from et4_filtered_stocks_market_cap import selected_stocks

CACHE_DIR = "data_cache"
INDICATORS_DIR = "main_indicators"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)

api_semaphore = threading.Semaphore(2)

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = TimedRotatingFileHandler(
    "trading_script_signal1.log",
    when="M",
    interval=30,  # rotate logs every 30 minutes
    backupCount=5,
    delay=True
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.WARNING)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.WARNING)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logging.warning("Logging with TimedRotatingFileHandler.")
print("Script start...")

cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
try:
    os.chdir(cwd)
    logger.info(f"Changed working directory to {cwd}.")
except Exception as e:
    logger.error(f"Error changing dir: {e}")

market_holidays = [
    # ...
]

class APICallError(Exception):
    pass

SIGNALS_DB = "generated_signals.json"


def load_generated_signals():
    if os.path.exists(SIGNALS_DB):
        try:
            with open(SIGNALS_DB, 'r') as f:
                return set(json.load(f))
        except json.JSONDecodeError:
            logging.error("Signals DB corrupted.")
            return set()
    else:
        return set()

def save_generated_signals(generated_signals):
    with open(SIGNALS_DB, 'w') as f:
        json.dump(list(generated_signals), f)

def normalize_time(df, tz='Asia/Kolkata'):
    df = df.copy()
    if 'date' not in df.columns:
        raise KeyError("No date col.")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)

    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('UTC')
    df['date'] = df['date'].dt.tz_convert(tz)
    return df

def load_and_normalize_csv(file_path, expected_cols=None, tz='Asia/Kolkata'):
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=expected_cols if expected_cols else [])
    df = pd.read_csv(file_path)
    if 'date' in df.columns:
        df = normalize_time(df, tz)
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        if expected_cols:
            for c in expected_cols:
                if c not in df.columns:
                    df[c] = ""
    if expected_cols:
        for c in expected_cols:
            if c not in df.columns:
                df[c] = ""
        df = df[expected_cols]
    return df

def setup_kite_session():
    try:
        with open("access_token.txt",'r') as tf:
            access_token = tf.read().strip()
        with open("api_key.txt",'r') as kf:
            key_secret = kf.read().split()
        kite_obj = KiteConnect(api_key=key_secret[0])
        kite_obj.set_access_token(access_token)
        logging.info("Kite session OK.")
        print("Kite session OK.")
        return kite_obj
    except Exception as e:
        logging.error(f"Kite setup error: {e}")
        raise

kite = setup_kite_session()

def get_tokens_for_stocks(stocks):
    try:
        logging.info("Fetching tokens.")
        instrument_dump = kite.instruments("NSE")
        instrument_df = pd.DataFrame(instrument_dump)
        instrument_df['tradingsymbol'] = instrument_df['tradingsymbol'].str.upper()
        stocks_upper = [s.upper() for s in stocks]
        tokens = instrument_df[instrument_df['tradingsymbol'].isin(stocks_upper)][['tradingsymbol','instrument_token']]
        logging.info(f"Got tokens for {len(tokens)} stocks.")
        print(f"Tokens: {len(tokens)}")
        return dict(zip(tokens['tradingsymbol'], tokens['instrument_token']))
    except Exception as e:
        logging.error(f"Error fetch tokens: {e}")
        raise

shares_tokens = get_tokens_for_stocks(selected_stocks)


def build_signal_id(ticker, row):
    dt_str = row['date'].isoformat()
    return f"{ticker}-{dt_str}"

def detect_signals_in_memory(ticker, df_today, existing_signal_ids):
    signals_detected = []
    if df_today.empty:
        return signals_detected
    try:
        first_open = df_today['open'].iloc[0]
        latest_close= df_today['close'].iloc[-1]
        daily_change= ((latest_close - first_open)/ first_open)*100
    except:
        return signals_detected

    latest_row= df_today.iloc[-1]
    adx_val = latest_row.get('ADX',0)
    if adx_val <= 20:
        return signals_detected

    # Bullish or Bearish
    if (
        daily_change>1.25
        and latest_row.get("RSI",0)>35
        and latest_row.get("close",0)> latest_row.get("VWAP",0)*1.02
    ):
        trend_type= "Bullish"
    elif (
        daily_change< -1.25
        and latest_row.get("RSI",0)<65
        and latest_row.get("close",0)< latest_row.get("VWAP",0)*0.98
    ):
        trend_type= "Bearish"
    else:
        return signals_detected

    volume_multiplier=1.5
    rolling_window=20
    MACD_BULLISH_OFFSET=0.5
    MACD_BEARISH_OFFSET=-0.5
    MACD_BULLISH_OFFSETp=-1.0
    MACD_BEARISH_OFFSETp=1.0

    if trend_type=="Bullish":
        conditions= {
            "Pullback":(
                (df_today["low"]>= df_today["VWAP"]*0.95)
                & (df_today["close"]> df_today["open"])
                & (df_today["MACD"]>df_today["Signal_Line"])
                & (df_today["MACD"]>MACD_BULLISH_OFFSETp)
                & (df_today["close"]>df_today["20_SMA"])
                & (df_today["Stochastic"]<60)
                & (df_today["Stochastic"].diff()>0)
                & (df_today["ADX"]>20)
            ),
            "Breakout":(
                (df_today["close"]> df_today["high"].rolling(window=rolling_window).max()*0.97)
                & (df_today["volume"]> volume_multiplier* df_today["volume"].rolling(window=rolling_window).mean())
                & (df_today["MACD"]> df_today["Signal_Line"])
                & (df_today["MACD"]> MACD_BULLISH_OFFSET)
                & (df_today["close"]> df_today["20_SMA"])
                & (df_today["Stochastic"]>60)
                & (df_today["Stochastic"].diff()>0)
                & (df_today["ADX"]>30)
            ),
        }
    else:
        # Bearish
        conditions= {
            "Pullback":(
                (df_today["high"]<= df_today["VWAP"]*1.05)
                & (df_today["close"]< df_today["open"])
                & (df_today["MACD"]< df_today["Signal_Line"])
                & (df_today["MACD"]< MACD_BEARISH_OFFSETp)
                & (df_today["close"]< df_today["20_SMA"])
                & (df_today["Stochastic"]> 40)
                & (df_today["Stochastic"].diff()<0)
                & (df_today["ADX"]> 20)
            ),
            "Breakdown":(
                (df_today["close"]< df_today["low"].rolling(window=rolling_window).min()*1.03)
                & (df_today["volume"]> volume_multiplier* df_today["volume"].rolling(window=rolling_window).mean())
                & (df_today["MACD"]< df_today["Signal_Line"])
                & (df_today["MACD"]< MACD_BEARISH_OFFSET)
                & (df_today["close"]< df_today["20_SMA"])
                & (df_today["Stochastic"]< 40)
                & (df_today["Stochastic"].diff()<0)
                & (df_today["ADX"]>30)
            ),
        }

    for entry_type,cond in conditions.items():
        subset= df_today.loc[cond]
        if not subset.empty:
            for _, row in subset.iterrows():
                sig_id= row.get("Signal_ID","")
                if not sig_id:
                    sig_id= build_signal_id(ticker, row)
                if sig_id in existing_signal_ids:
                    continue
                signals_detected.append({
                    "Ticker": ticker,
                    "date": row["date"],
                    "Entry Type": entry_type,
                    "Trend Type": trend_type,
                    "Price": round(row.get("close",0),2),
                    "Daily Change %": round(daily_change,2),
                    "VWAP": round(row.get("VWAP",0),2),
                    "ADX": round(row.get("ADX",0),2),
                    "MACD": row.get("MACD",None),
                    "Signal_ID": sig_id,
                    "logtime": row.get("logtime", datetime.now(india_tz).isoformat()),
                    "Entry Signal":"Yes"
                })
    return signals_detected

def get_last_trading_day(ref_date=None):
    if ref_date is None:
        ref_date= datetime.now(india_tz).date()
    if ref_date.weekday()<5 and ref_date not in market_holidays:
        return ref_date
    else:
        d= ref_date - timedelta(days=1)
        while d.weekday()>=5 or d in market_holidays:
            d-= timedelta(days=1)
        return d

def process_csv_with_retries(file_path, ticker, max_retries=3, delay=5):
    for attempt in range(1,max_retries+1):
        try:
            df= load_and_normalize_csv(file_path, tz='Asia/Kolkata')
            return df
        except Exception as e:
            logging.error(f"[{ticker}] CSV load error attempt {attempt}: {e}")
            if attempt<max_retries:
                time.sleep(delay)
            else:
                raise

def mark_signals_in_main_csv(ticker, signals_list, main_path, tz_obj):
    if not os.path.exists(main_path):
        logging.warning(f"[{ticker}] No {main_path}. Skipping.")
        return
    lock_path= main_path+".lock"
    lock= FileLock(lock_path, timeout=10)
    try:
        with lock:
            df_main = load_and_normalize_csv(main_path, tz=tz_obj)
            if df_main.empty:
                logging.warning(f"[{ticker}] main_indicators empty. Skip.")
                return
            if 'Entry Signal' not in df_main.columns:
                df_main['Entry Signal']='No'
            updated=0
            for signal in signals_list:
                sig_id= signal.get("Signal_ID","")
                sig_date=signal.get("date",None)
                if sig_id and 'Signal_ID' in df_main.columns:
                    mask= (df_main['Signal_ID']== sig_id)
                else:
                    mask= (df_main['date']== sig_date)
                if mask.any():
                    mask_up= mask&(df_main['Entry Signal']!='Yes')
                    if mask_up.any():
                        df_main.loc[mask_up,'Entry Signal']='Yes'
                        df_main.loc[mask_up,'logtime']= signal.get('logtime', datetime.now(tz_obj).isoformat())
                        updated+= mask_up.sum()
            if updated>0:
                df_main.sort_values('date', inplace=True)
                df_main.to_csv(main_path, index=False)
                logging.info(f"[{ticker}] Marked 'Yes' for {updated} signals.")
    except Timeout:
        logging.error(f"[{ticker}] Lock Timeout for {main_path}.")
    except Exception as e:
        logging.error(f"[{ticker}] Error marking signals: {e}")

def find_price_action_entries_for_today(last_trading_day):
    st= india_tz.localize(datetime.combine(last_trading_day, dt_time(9,25)))
    et= india_tz.localize(datetime.combine(last_trading_day, dt_time(14,50)))
    pattern= os.path.join(INDICATORS_DIR, '*_main_indicators.csv')
    files= glob.glob(pattern)
    if not files:
        print("No main_indicators CSV found.")
        return pd.DataFrame()

    all_signals=[]
    gen_sigs= load_generated_signals()
    for file_path in tqdm(files, desc="Process CSV", unit="file"):
        ticker= os.path.basename(file_path).replace('_main_indicators.csv','').upper()
        try:
            df= process_csv_with_retries(file_path, ticker)
            df_today= df[(df['date']>= st)&(df['date']<= et)].copy()
            if df_today.empty:
                continue
            new_signals= detect_signals_in_memory(ticker, df_today, gen_sigs)
            if new_signals:
                for sig in new_signals:
                    gen_sigs.add(sig["Signal_ID"])
                all_signals.extend(new_signals)
        except:
            continue
    save_generated_signals(gen_sigs)
    if not all_signals:
        return pd.DataFrame()
    signals_df= pd.DataFrame(all_signals)
    signals_df.sort_values('date', inplace=True)
    signals_df.reset_index(drop=True, inplace=True)
    return signals_df

def create_papertrade_file(input_file, output_file, last_trading_day):
    if not os.path.exists(input_file):
        print("No signals file, skip papertrade.")
        return
    try:
        df_entries= pd.read_csv(input_file)
        if 'date' not in df_entries.columns:
            raise KeyError("No date col.")
        df_entries= normalize_time(df_entries)
        df_entries.sort_values('date', inplace=True)

        st= india_tz.localize(datetime.combine(last_trading_day, dt_time(9,25)))
        et= india_tz.localize(datetime.combine(last_trading_day, dt_time(14,50)))
        todays_entries= df_entries[(df_entries['date']>= st)&(df_entries['date']<= et)].copy()
        if todays_entries.empty:
            return
        todays_entries.sort_values('date', inplace=True)
        todays_entries.drop_duplicates(subset=['Ticker'], keep='first', inplace=True)

        lock_path= output_file+".lock"
        lock= FileLock(lock_path, timeout=10)
        with lock:
            if os.path.exists(output_file):
                existing= pd.read_csv(output_file)
                existing= normalize_time(existing)
                combined= pd.concat([existing, todays_entries], ignore_index=True)
                combined.drop_duplicates(subset=['Ticker','date'], keep='last', inplace=True)
                combined.sort_values('date', inplace=True)
                combined.to_csv(output_file, index=False)
            else:
                todays_entries.to_csv(output_file, index=False)
        print(f"Papertrade => {output_file}")
    except Exception as e:
        print(f"create_papertrade error: {e}")
        logging.error(f"papertrade error: {e}")

def main():
    last_trading_day= get_last_trading_day()
    today_str= last_trading_day.strftime('%Y-%m-%d')
    entries_file= f"price_action_entries_15min_{today_str}.csv"
    papertrade_file= f"papertrade_{today_str}.csv"

    signals_df= find_price_action_entries_for_today(last_trading_day)
    if signals_df.empty:
        print("No signals found this cycle.")
        return

    header_needed= not os.path.exists(entries_file)
    # append new signals each cycle
    signals_df.to_csv(entries_file, mode='a', index=False, header=header_needed)
    print(f"Saved {len(signals_df)} signals => {entries_file}")

    # mark signals in main CSV
    for row in signals_df.to_dict('records'):
        ticker= row["Ticker"]
        main_path= os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
        mark_signals_in_main_csv(ticker, [row], main_path, india_tz)

    print("Updated 'Entry Signal' in main_indicators CSV.")

    # create or append to papertrade
    create_papertrade_file(entries_file, papertrade_file, last_trading_day)
    print(f"Papertrade => {papertrade_file}")

def signal_handler(sig, frame):
    print("Interrupt received, shutting down.")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ================================
# Run every 30 seconds
# ================================
def schedule_jobs():
    """
    Now we run main() every 30 seconds to continuously check for
    newly updated rows in the CSV.
    Because updates to main_indicators.csv might come in over 4 mins,
    multiple 30s checks ensure we won't miss newly appended data.
    """
    schedule.every(30).seconds.do(main)

def run_schedule_loop():
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__=="__main__":
    schedule_jobs()
    run_schedule_loop()
