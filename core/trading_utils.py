from abc import ABC, abstractmethod
from dataclasses import dataclass
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.graph_objects as go
import time
import requests.exceptions


class StockDataFetcher:
    def __init__(self, symbol, start_date, end_date, interval='1d'):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.data = None

    def fetch_data(self, max_retries=3, delay=2):
        """Fetch data with retry logic for timeout issues"""
        for attempt in range(max_retries):
            try:
                print(f"Descargando {self.symbol} (intento {attempt + 1}/{max_retries})...")
                df = yf.download(
                    self.symbol, 
                    start=self.start_date, 
                    end=self.end_date, 
                    interval=self.interval,
                    progress=False,  # Reducir output
                    timeout=30,  # Timeout más largo
                    auto_adjust=True  # Explícitamente especificar
                )
                
                if df.empty:
                    raise ValueError(f"No data found for {self.symbol}")
                
                # Limpiar columnas si es MultiIndex
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                df.index = pd.to_datetime(df.index)
                df = df.reset_index()
                df.rename(columns={'Date': 'Datetime'}, inplace=True, errors='ignore')
                self.data = df
                print(f"✅ {self.symbol} descargado exitosamente")
                return df
                
            except (requests.exceptions.Timeout, Exception) as e:
                print(f"❌ Error descargando {self.symbol}: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Reintentando en {delay} segundos...")
                    time.sleep(delay)
                else:
                    print(f"❌ Falló descarga de {self.symbol} después de {max_retries} intentos")
                    return pd.DataFrame() 

class EventDetector:
    @staticmethod
    def detect_local_extrema(df, order=10):
        df['min_local'] = df.loc[signal.argrelextrema(df['Low'].values, np.less, order=order),'Low']
        df['max_local'] = df.loc[signal.argrelextrema(df['High'].values, np.greater, order=order),'High']
        return df

    @staticmethod
    def detect_high_volume_iqr(df, window=20, multiplier=1.5):
        q1 = df['Volume'].rolling(window=window).quantile(0.25)
        q3 = df['Volume'].rolling(window=window).quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + multiplier * iqr
        df['high_volume'] = df['Volume'] > upper_bound
        return df

    @staticmethod
    def detect_price_variation(df, price_threshold=0.02):
        df['price_variation'] = (df['Close'].diff().abs() / df['Close'].shift(1).abs()) > price_threshold
        df['price_variation'] = df['price_variation'].fillna(False)
        return df

    @staticmethod
    def avwap(df, date):
        mask = df['Datetime'] >= date
        vol = df.loc[mask, 'Volume']
        price = df.loc[mask, ['High', 'Low', 'Close']].mean(axis=1)

        avwap_values = (price * vol).cumsum() / vol.cumsum()
        return avwap_values

    @staticmethod
    def detect_key_events_and_avwap(df):
        key_events = df[(df['min_local'].notna() | df['max_local'].notna()) & (df['high_volume'] | df['price_variation'])]
        for index, row in key_events.iterrows():
            date = df.loc[index, 'Datetime'].strftime('%Y-%m-%d %H:%M:%S')
            df[f'AVWAP_{date}'] = EventDetector.avwap(df.copy(), date)
        return df

class Visualizer:
    @staticmethod
    def plot_key_events(df, symbol):
        plt.figure(figsize=(12,6))
        plt.plot(df['Datetime'], df['Close'], label='Close Price', color='black')
        plt.scatter(df['Datetime'], df['min_local'], color='green', label='Min Local', marker='^', alpha=0.7)
        plt.scatter(df['Datetime'], df['max_local'], color='red', label='Max Local', marker='v', alpha=0.7)
        plt.scatter(df['Datetime'][df['high_volume']], df['Close'][df['high_volume']], color='blue', label='High Volume', marker='o', alpha=0.7)
        plt.scatter(df['Datetime'][df['price_variation']], df['Close'][df['price_variation']], color='violet', label='Price Variation', marker='o', alpha=0.7)
        plt.legend()
        plt.xlabel('Datetime')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.title(f'{symbol} Key Event Detection')
        plt.show()

    @staticmethod
    def plot_avwap_lines(df, symbol):
        plt.figure(figsize=(12, 6))
        plt.plot(df['Datetime'], df['Close'], label='Close Price', color='black')
        avwap_columns = [col for col in df.columns if col.startswith('AVWAP_') ]
        for col in avwap_columns:
            plt.plot(df['Datetime'], df[col], label=col.replace('AVWAP_', 'AVWAP ('), alpha=0.7)
        plt.legend()
        plt.title(f'{symbol} Close Price with AVWAP Lines')
        plt.xlabel('Datetime')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()


class Backtest:
    def __init__(self, trades_df, fetcher, initial_balance=10000, risk_reward_ratio=2, risk_balance=0.01, commission=0.0005, spread=0.0002):
        self.trades_df = trades_df
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.balance_over_time = []
        self.trades_executed = 0
        self.win_trades = 0
        self.loss_trades = 0
        self.risk_reward_ratio = risk_reward_ratio
        self.risk_balance = risk_balance
        self.fetcher = fetcher
        self.commission = commission
        self.spread = spread
        self.max_drawdown = 0
        self.peak_balance = initial_balance
        self.trade_results = []
        self.price_data = self.fetcher.fetch_data()

    def run_backtest(self):
        if self.price_data.empty:
            print(f"❌ No hay datos disponibles para {self.fetcher.symbol}")
            return
            
        price_data = self.price_data
        balance = self.initial_balance
        balance_history = []
        trade_results = []

        peak_balance = balance
        max_drawdown = 0

        for _, trade in self.trades_df.iterrows():
            if balance <= 0:
                print("❌ Balance agotado. Backtest detenido.")
                break

            risk_per_trade = balance * self.risk_balance

            entry_price = trade.Entry
            tp = trade.TakeProfit
            sl = trade.StopLoss
            direction = trade.Type
            entry_time = trade.Datetime

            # Aplicar spread
            if direction == 'long':
                entry_price += self.spread
                tp += self.spread
                sl += self.spread
            else:
                entry_price -= self.spread
                tp -= self.spread
                sl -= self.spread

            trade_data = price_data[price_data['Datetime'] >= entry_time]

            hit_tp, hit_sl = False, False
            exit_time, exit_price = None, None

            for _, row in trade_data.iterrows():
                if direction == 'long' and row['Low'] <= sl:
                    hit_sl = True
                    exit_price = sl
                    exit_time = row['Datetime']
                    break
                if direction == 'long' and row['High'] >= tp:
                    hit_tp = True
                    exit_price = tp
                    exit_time = row['Datetime']
                    break
                if direction == 'short' and row['High'] >= sl:
                    hit_sl = True
                    exit_price = sl
                    exit_time = row['Datetime']
                    break
                if direction == 'short' and row['Low'] <= tp:
                    hit_tp = True
                    exit_price = tp
                    exit_time = row['Datetime']
                    break

            pnl = risk_per_trade * self.risk_reward_ratio if hit_tp else -risk_per_trade
            pnl -= abs(entry_price * self.commission)
            balance += pnl

            self.trades_executed += 1
            self.win_trades += int(hit_tp)
            self.loss_trades += int(hit_sl)
            balance_history.append(balance)

            trade_results.append({
                "Datetime": entry_time,  # Cambiado de EntryTime
                "Entry": entry_price,    # Cambiado de EntryPrice
                "ExitTime": exit_time,
                "Exit": exit_price,      # Cambiado de ExitPrice
                "Type": direction,       # Cambiado de Direction
                "Result": "TP" if hit_tp else "SL",
                "PnL": pnl
            })

            peak_balance = max(peak_balance, balance)
            drawdown = (peak_balance - balance) / peak_balance
            max_drawdown = max(max_drawdown, drawdown)

        self.trade_results = pd.DataFrame(trade_results)
        self.balance_over_time = balance_history
        self.balance = balance
        self.max_drawdown = max_drawdown
        self.peak_balance = peak_balance

    def plot_performance(self, return_fig=True):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=self.balance_over_time, mode='lines', name='Balance'))
        fig.add_trace(go.Scatter(y=[self.initial_balance] * len(self.balance_over_time), mode='lines', name='Initial Balance', line=dict(dash='dash')))
        fig.update_layout(title='Backtest Performance', xaxis_title='Trades', yaxis_title='Balance', template='plotly_dark')
        
        if return_fig:
            return fig
        else:
            fig.show()

    def plot_trades(self, return_fig=True):
        if self.price_data.empty or self.trade_results.empty:
            # Crear gráfico vacío si no hay datos
            fig = go.Figure()
            fig.add_annotation(
                text="No hay datos de trades disponibles",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig if return_fig else fig.show()
            
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=self.price_data['Datetime'],
            open=self.price_data['Open'],
            high=self.price_data['High'],
            low=self.price_data['Low'],
            close=self.price_data['Close'],
            name='Price'))

        for index, trade in self.trade_results.iterrows():
            # Verificar si las columnas existen antes de usarlas
            exit_type = trade.get('Result', 'Unknown')
            color = 'green' if exit_type == 'TP' else 'red'
            
            # Usar las columnas correctas
            entry_time = trade.get('Datetime', trade.get('EntryTime'))
            exit_time = trade.get('ExitTime')
            entry_price = trade.get('Entry', trade.get('EntryPrice'))
            exit_price = trade.get('Exit', trade.get('ExitPrice'))
            trade_type = trade.get('Type', trade.get('Direction', 'Unknown'))
            
            if pd.notna(entry_time) and pd.notna(entry_price):
                fig.add_trace(go.Scatter(
                    x=[entry_time, exit_time] if pd.notna(exit_time) else [entry_time],
                    y=[entry_price, exit_price] if pd.notna(exit_price) else [entry_price],
                    mode='markers+lines' if pd.notna(exit_time) else 'markers',
                    marker=dict(color=color, size=10),
                    name=f"Trade {trade_type}",
                    showlegend=False
                ))

        fig.update_layout(title='Trade Entries & Exits', xaxis_title='Datetime', yaxis_title='Price')
        if return_fig:
            return fig
        else:
            fig.show()

    def print_summary(self):
        print(f"Initial Balance: {self.initial_balance}")
        print(f"Final Balance: {self.balance}")
        print(f"Total Trades Executed: {self.trades_executed}")
        print(f"Winning Trades: {self.win_trades}")
        print(f"Losing Trades: {self.loss_trades}")
        win_rate = (self.win_trades / self.trades_executed) * 100 if self.trades_executed > 0 else 0
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Max Drawdown: {self.max_drawdown:.2%}")

@dataclass
class Trade:
    Datetime: pd.Timestamp
    Entry: float
    TakeProfit: float
    StopLoss: float
    Type: str  # 'long' o 'short'

class TradingStrategy(ABC):
    def __init__(self, strategy_name, risk_reward_ratio=2, stop_loss_pct=0.02):
        self.strategy_name = strategy_name
        self.risk_reward_ratio = risk_reward_ratio
        self.stop_loss_pct = stop_loss_pct

    @abstractmethod
    def generate_signals(self, df):
        pass

    def apply_strategy(self, df) -> pd.DataFrame:
        df = self.generate_signals(df)
        trades = []

        for _, row in df.iterrows():
            if row.get('long_signal', False):
                entry = row['Close']
                stop_loss = entry * (1 - self.stop_loss_pct)
                take_profit = entry + (entry - stop_loss) * self.risk_reward_ratio
                trades.append(Trade(row['Datetime'], entry, take_profit, stop_loss, 'long'))

            if row.get('short_signal', False):
                entry = row['Close']
                stop_loss = entry * (1 + self.stop_loss_pct)
                take_profit = entry - (stop_loss - entry) * self.risk_reward_ratio
                trades.append(Trade(row['Datetime'], entry, take_profit, stop_loss, 'short'))

        return pd.DataFrame(trades)

