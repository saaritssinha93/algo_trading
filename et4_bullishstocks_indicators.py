import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import logging
import ta  # For technical analysis indicators

# Importing the shares list from et1_stock_tickers.py
from et4_nfo_stock_tickers import shares

# Define the correct path
cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
os.chdir(cwd)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bullish_stocks.log"),
        logging.StreamHandler()
    ]
)

# Function to setup Kite Connect session
def setup_kite_session():
    try:
        with open("access_token.txt", 'r') as token_file, open("api_key.txt", 'r') as key_file:
            api_key = key_file.read().split()[0]
            access_token = token_file.read().strip()
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        logging.info("Kite session established successfully.")
        return kite
    except Exception as e:
        logging.error(f"Error setting up Kite session: {e}")
        raise

# Fetch historical data
def fetch_historical_data(kite, token, interval, days):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        data = pd.DataFrame(kite.historical_data(token, start_date, end_date, interval))
        if data.empty:
            logging.warning(f"No data returned for token: {token}")
            return pd.DataFrame()
        data['date'] = pd.to_datetime(data['date'])
        return data.set_index('date')
    except Exception as e:
        logging.error(f"Error fetching historical data for token {token}: {e}")
        return pd.DataFrame()

# Calculate technical indicators
def calculate_indicators(df):
    if df.empty:
        return pd.DataFrame()
    try:
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['EMA_10'] = df['close'].ewm(span=10).mean()
        df['EMA_50'] = df['close'].ewm(span=50).mean()
        df['MACD'] = ta.trend.MACD(df['close']).macd()
        df['Signal_Line'] = ta.trend.MACD(df['close']).macd_signal()
        df['ADX'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        return df
    except Exception as e:
        logging.error(f"Error calculating indicators: {e}")
        return pd.DataFrame()

# Bullish criteria for intraday
def is_bullish_intraday(df):
    if df.empty:
        return False
    latest = df.iloc[-1]
    return (
        latest['RSI'] > 50 and
        latest['EMA_10'] > latest['EMA_50'] and
        latest['MACD'] > latest['Signal_Line']
    )

# Bullish criteria for swing trading
def is_bullish_swing(df):
    if df.empty:
        return False
    latest = df.iloc[-1]
    return (
        latest['RSI'] > 50 and
        latest['ADX'] > 20 and
        latest['EMA_10'] > latest['EMA_50']
    )

# Main function to filter stocks
def filter_bullish_stocks(kite, shares, interval="15minute", days=5, mode="intraday"):
    bullish_stocks = []
    instrument_dump = kite.instruments("NSE")
    instrument_df = pd.DataFrame(instrument_dump)

    for stock in shares:
        try:
            token = instrument_df[instrument_df['tradingsymbol'] == stock]['instrument_token'].values[0]
            data = fetch_historical_data(kite, token, interval, days)
            data = calculate_indicators(data)

            if mode == "intraday" and is_bullish_intraday(data):
                bullish_stocks.append(stock)
            elif mode == "swing" and is_bullish_swing(data):
                bullish_stocks.append(stock)
        
        except Exception as e:
            logging.error(f"Error processing {stock}: {e}")
    
    return bullish_stocks

# Main execution
if __name__ == "__main__":
    try:
        # Set up Kite Connect session
        kite = setup_kite_session()

        # Filter intraday bullish stocks
        intraday_bullish = filter_bullish_stocks(kite, shares, interval="15minute", days=5, mode="intraday")
        print("Intraday Bullish Stocks:", intraday_bullish)
        logging.info(f"Intraday Bullish Stocks: {intraday_bullish}")
        
        # Filter swing trading bullish stocks
        swing_bullish = filter_bullish_stocks(kite, shares, interval="day", days=30, mode="swing")
        print("Swing Trading Bullish Stocks:", swing_bullish)
        logging.info(f"Swing Trading Bullish Stocks: {swing_bullish}")
    
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
