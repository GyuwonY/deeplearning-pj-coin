
import requests
import csv
from datetime import datetime, timedelta
import time
import os

def get_krw_markets():    
    daycandle_dir = 'daycandle'
    markets = [f.replace('.csv', '') for f in os.listdir(daycandle_dir) if f.endswith('.csv')]    
    return markets

def get_candles(market, to_time):
    url = "https://api.upbit.com/v1/candles/days"
    params = {
        'market': market,
        'count': 200,
        'to': to_time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    headers = {"accept": "application/json"}
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"  [Error] Could not fetch candles for {market}: {e}")
        return None
    except ValueError: # Catches JSON decoding errors
        print(f"  [Error] Could not decode JSON for {market}. Response: {response.text}")
        return None


def save_to_csv(market, candles):
    # Ensure the data directory exists
    if not os.path.exists('daycandle'):
        os.makedirs('daycandle')
        
    filename = f"daycandle/{market}.csv"
    is_new_file = not os.path.exists(filename)

    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['market', 'candle_date_time_utc', 'opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_price', 'candle_acc_trade_volume', 'change_rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if is_new_file:
            writer.writeheader()

        for candle in candles:
            if not isinstance(candle, dict):
                continue
            writer.writerow({
                'market': candle.get('market'),
                'candle_date_time_utc': candle.get('candle_date_time_utc'),
                'opening_price': candle.get('opening_price'),
                'high_price': candle.get('high_price'),
                'low_price': candle.get('low_price'),
                'trade_price': candle.get('trade_price'),
                'candle_acc_trade_price': candle.get('candle_acc_trade_price'),
                'candle_acc_trade_volume': candle.get('candle_acc_trade_volume'),
                'change_rate': candle.get('change_rate')
            })

if __name__ == "__main__":
    # Clear existing csv files before starting
    # if os.path.exists('data'):
    #     for item in os.listdir('data'):
    #         if item.endswith(".csv"):
    #             os.remove(os.path.join('data', item))

    krw_markets = get_krw_markets()
    
    if krw_markets:
        for market in krw_markets:
            print(f"Fetching data for {market}...")
            to_time = datetime(2025, 7, 30, 0, 0, 0)
            
            while True:
                candles = get_candles(market, to_time)
                time.sleep(0.11) # To avoid rate limiting

                if not candles: # This now handles None or empty list
                    break

                save_to_csv(market, candles)

                if len(candles) < 200:
                    break

                # Ensure the last candle is a dictionary before accessing it
                last_candle = candles[-1]
                if not isinstance(last_candle, dict) or 'candle_date_time_utc' not in last_candle:
                    print(f"  [Error] Invalid last candle format for {market}. Stopping.")
                    break

                last_candle_time_str = last_candle['candle_date_time_utc']
                last_candle_time = datetime.strptime(last_candle_time_str, '%Y-%m-%dT%H:%M:%S')
                to_time = last_candle_time
            print(f"Finished fetching data for {market}")
