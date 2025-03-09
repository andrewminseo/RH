import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import robin_stocks.robinhood as rh
import datetime
import json
from auth import login_with_credentials
import time
from pathlib import Path

class StockAnalyzer:
    def __init__(self):
        """Initialize the stock analyzer with authentication"""
        self.login = login_with_credentials()
        self.data_dir = Path("stock_data")
        self.data_dir.mkdir(exist_ok=True)
        self.results_dir = Path("analysis_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def get_stock_data(self, symbols, interval="day", span="year"):
        """
        Get historical data for a list of stock symbols
        
        Parameters:
        symbols (list): List of stock symbols
        interval (str): 'day', '5minute', etc.
        span (str): 'day', 'week', 'month', '3month', 'year', '5year'
        
        Returns:
        dict: Dictionary with symbols as keys and DataFrames as values
        """
        stock_data = {}
        
        for symbol in symbols:
            try:
                print(f"Fetching data for {symbol}...")
                historicals = rh.stocks.get_stock_historicals(
                    symbol, 
                    interval=interval, 
                    span=span
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(historicals)
                
                # Convert columns to appropriate types
                df['begins_at'] = pd.to_datetime(df['begins_at'])
                for col in ['open_price', 'close_price', 'high_price', 'low_price', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                
                stock_data[symbol] = df
                
                # Save to CSV
                csv_path = self.data_dir / f"{symbol}_{interval}_{span}.csv"
                df.to_csv(csv_path, index=False)
                print(f"Data for {symbol} saved to {csv_path}")
                
                # Don't overwhelm the API
                time.sleep(1)
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        return stock_data
    
    def get_fundamentals(self, symbols):
        """
        Get fundamental data for a list of stock symbols
        
        Parameters:
        symbols (list): List of stock symbols
        
        Returns:
        dict: Dictionary with symbols as keys and fundamental data as values
        """
        fundamentals = {}
        
        for symbol in symbols:
            try:
                print(f"Fetching fundamentals for {symbol}...")
                fundamental_data = rh.stocks.get_fundamentals(symbol)
                
                # If we get a list, take the first item
                if isinstance(fundamental_data, list) and len(fundamental_data) > 0:
                    fundamental_data = fundamental_data[0]
                
                fundamentals[symbol] = fundamental_data
                
                # Save to JSON
                json_path = self.data_dir / f"{symbol}_fundamentals.json"
                with open(json_path, 'w') as f:
                    json.dump(fundamental_data, f, indent=4)
                print(f"Fundamentals for {symbol} saved to {json_path}")
                
                # Don't overwhelm the API
                time.sleep(1)
                
            except Exception as e:
                print(f"Error fetching fundamentals for {symbol}: {e}")
        
        return fundamentals
    
    def analyze_volatility(self, stock_data):
        """
        Analyze volatility for stock data
        
        Parameters:
        stock_data (dict): Dictionary with symbols as keys and DataFrames as values
        
        Returns:
        pd.DataFrame: DataFrame with volatility metrics
        """
        volatility_results = []
        
        for symbol, df in stock_data.items():
            try:
                # Calculate daily returns
                df['daily_return'] = df['close_price'].pct_change()
                
                # Calculate metrics
                daily_volatility = df['daily_return'].std()
                annualized_volatility = daily_volatility * (252 ** 0.5)  # Assuming 252 trading days
                
                # High-Low Range volatility
                df['hl_range'] = (df['high_price'] - df['low_price']) / df['low_price']
                range_volatility = df['hl_range'].mean()
                
                volatility_results.append({
                    'Symbol': symbol,
                    'Daily_Volatility': daily_volatility,
                    'Annualized_Volatility': annualized_volatility,
                    'Range_Volatility': range_volatility
                })
                
            except Exception as e:
                print(f"Error analyzing volatility for {symbol}: {e}")
        
        # Convert to DataFrame
        volatility_df = pd.DataFrame(volatility_results)
        
        # Save to CSV
        csv_path = self.results_dir / "volatility_analysis.csv"
        volatility_df.to_csv(csv_path, index=False)
        print(f"Volatility analysis saved to {csv_path}")
        
        return volatility_df
    
    def analyze_moving_averages(self, stock_data, short_window=20, long_window=50):
        """
        Calculate moving averages and generate signals
        
        Parameters:
        stock_data (dict): Dictionary with symbols as keys and DataFrames as values
        short_window (int): Short moving average window
        long_window (int): Long moving average window
        
        Returns:
        dict: Dictionary with symbols as keys and signal DataFrames as values
        """
        signals = {}
        
        for symbol, df in stock_data.items():
            try:
                # Create a copy to avoid modifying the original
                signals_df = df.copy()
                
                # Compute short and long moving averages
                signals_df['short_ma'] = signals_df['close_price'].rolling(window=short_window, min_periods=1).mean()
                signals_df['long_ma'] = signals_df['close_price'].rolling(window=long_window, min_periods=1).mean()
                
                # Create signals
                signals_df['signal'] = 0.0
                signals_df['signal'][short_window:] = np.where(
                    signals_df['short_ma'][short_window:] > signals_df['long_ma'][short_window:], 1.0, 0.0
                )
                
                # Generate trading orders
                signals_df['position'] = signals_df['signal'].diff()
                
                signals[symbol] = signals_df
                
                # Save to CSV
                csv_path = self.results_dir / f"{symbol}_ma_signals.csv"
                signals_df.to_csv(csv_path, index=False)
                print(f"Moving average signals for {symbol} saved to {csv_path}")
                
                # Create and save plot
                plt.figure(figsize=(12, 6))
                plt.plot(signals_df['begins_at'], signals_df['close_price'], label='Close Price')
                plt.plot(signals_df['begins_at'], signals_df['short_ma'], label=f'{short_window} Day MA')
                plt.plot(signals_df['begins_at'], signals_df['long_ma'], label=f'{long_window} Day MA')
                
                # Plot buy/sell signals
                plt.plot(signals_df.loc[signals_df['position'] == 1.0].begins_at, 
                         signals_df.loc[signals_df['position'] == 1.0].close_price, 
                         '^', markersize=10, color='g', lw=0, label='Buy Signal')
                plt.plot(signals_df.loc[signals_df['position'] == -1.0].begins_at, 
                         signals_df.loc[signals_df['position'] == -1.0].close_price, 
                         'v', markersize=10, color='r', lw=0, label='Sell Signal')
                
                plt.title(f'{symbol} Moving Average Crossover')
                plt.xlabel('Date')
                plt.ylabel('Price ($)')
                plt.legend()
                plt.grid(True)
                
                plt_path = self.results_dir / f"{symbol}_ma_signals.png"
                plt.savefig(plt_path)
                plt.close()
                print(f"Moving average plot for {symbol} saved to {plt_path}")
                
            except Exception as e:
                print(f"Error analyzing moving averages for {symbol}: {e}")
        
        return signals
    
    def analyze_rsi(self, stock_data, window=14):
        """
        Calculate Relative Strength Index (RSI)
        
        Parameters:
        stock_data (dict): Dictionary with symbols as keys and DataFrames as values
        window (int): RSI calculation window
        
        Returns:
        dict: Dictionary with symbols as keys and RSI DataFrames as values
        """
        rsi_results = {}
        
        for symbol, df in stock_data.items():
            try:
                # Create a copy to avoid modifying the original
                rsi_df = df.copy()
                
                # Calculate price changes
                rsi_df['price_change'] = rsi_df['close_price'].diff()
                
                # Get gain and loss
                rsi_df['gain'] = rsi_df['price_change'].clip(lower=0)
                rsi_df['loss'] = rsi_df['price_change'].clip(upper=0).abs()
                
                # Calculate average gain and loss
                rsi_df['avg_gain'] = rsi_df['gain'].rolling(window=window, min_periods=1).mean()
                rsi_df['avg_loss'] = rsi_df['loss'].rolling(window=window, min_periods=1).mean()
                
                # Calculate RS and RSI
                rsi_df['rs'] = rsi_df['avg_gain'] / rsi_df['avg_loss']
                rsi_df['rsi'] = 100 - (100 / (1 + rsi_df['rs']))
                
                rsi_results[symbol] = rsi_df
                
                # Save to CSV
                csv_path = self.results_dir / f"{symbol}_rsi.csv"
                rsi_df.to_csv(csv_path, index=False)
                print(f"RSI analysis for {symbol} saved to {csv_path}")
                
                # Create and save plot
                plt.figure(figsize=(12, 8))
                
                # Price subplot
                ax1 = plt.subplot(211)
                ax1.plot(rsi_df['begins_at'], rsi_df['close_price'])
                ax1.set_title(f'{symbol} Close Price')
                ax1.grid(True)
                
                # RSI subplot
                ax2 = plt.subplot(212, sharex=ax1)
                ax2.plot(rsi_df['begins_at'], rsi_df['rsi'], color='purple')
                ax2.axhline(70, color='red', linestyle='--')
                ax2.axhline(30, color='green', linestyle='--')
                ax2.fill_between(rsi_df['begins_at'], y1=30, y2=70, alpha=0.1, color='gray')
                ax2.set_title(f'{symbol} RSI ({window} periods)')
                ax2.set_ylim(0, 100)
                ax2.grid(True)
                
                plt.tight_layout()
                
                plt_path = self.results_dir / f"{symbol}_rsi.png"
                plt.savefig(plt_path)
                plt.close()
                print(f"RSI plot for {symbol} saved to {plt_path}")
                
            except Exception as e:
                print(f"Error analyzing RSI for {symbol}: {e}")
        
        return rsi_results
    
    def analyze_correlation(self, stock_data):
        """
        Analyze correlation between different stocks
        
        Parameters:
        stock_data (dict): Dictionary with symbols as keys and DataFrames as values
        
        Returns:
        pd.DataFrame: Correlation matrix
        """
        if len(stock_data) < 2:
            print("Need at least 2 stocks to analyze correlation")
            return None
        
        try:
            # Create a DataFrame with close prices for all stocks
            close_prices = pd.DataFrame()
            
            for symbol, df in stock_data.items():
                close_prices[symbol] = df['close_price'].values
            
            # Calculate correlation matrix
            correlation_matrix = close_prices.corr()
            
            # Save to CSV
            csv_path = self.results_dir / "correlation_matrix.csv"
            correlation_matrix.to_csv(csv_path)
            print(f"Correlation matrix saved to {csv_path}")
            
            # Create and save heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
            plt.title('Stock Price Correlation Matrix')
            
            plt_path = self.results_dir / "correlation_heatmap.png"
            plt.savefig(plt_path)
            plt.close()
            print(f"Correlation heatmap saved to {plt_path}")
            
            return correlation_matrix
            
        except Exception as e:
            print(f"Error analyzing correlation: {e}")
            return None
    
    def generate_summary_report(self, symbols):
        """
        Generate a summary report of all analyses
        
        Parameters:
        symbols (list): List of stock symbols analyzed
        
        Returns:
        str: Path to the generated report
        """
        try:
            report_path = self.results_dir / "summary_report.md"
            
            with open(report_path, 'w') as f:
                f.write("# Stock Analysis Summary Report\n\n")
                f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## Stocks Analyzed\n\n")
                for symbol in symbols:
                    f.write(f"- {symbol}\n")
                
                f.write("\n## Analysis Results\n\n")
                
                # Volatility summary
                if os.path.exists(self.results_dir / "volatility_analysis.csv"):
                    f.write("### Volatility Analysis\n\n")
                    volatility_df = pd.read_csv(self.results_dir / "volatility_analysis.csv")
                    f.write(volatility_df.to_markdown(index=False))
                    f.write("\n\n")
                
                # Moving Average signals summary
                f.write("### Moving Average Signals\n\n")
                for symbol in symbols:
                    ma_path = self.results_dir / f"{symbol}_ma_signals.csv"
                    if os.path.exists(ma_path):
                        signals_df = pd.read_csv(ma_path)
                        buy_signals = (signals_df['position'] == 1.0).sum()
                        sell_signals = (signals_df['position'] == -1.0).sum()
                        
                        f.write(f"#### {symbol}\n\n")
                        f.write(f"- Buy Signals: {buy_signals}\n")
                        f.write(f"- Sell Signals: {sell_signals}\n\n")
                        f.write(f"![{symbol} Moving Average]({symbol}_ma_signals.png)\n\n")
                
                # RSI summary
                f.write("### RSI Analysis\n\n")
                for symbol in symbols:
                    rsi_path = self.results_dir / f"{symbol}_rsi.csv"
                    if os.path.exists(rsi_path):
                        rsi_df = pd.read_csv(rsi_path)
                        
                        # Calculate overbought/oversold conditions
                        overbought = (rsi_df['rsi'] > 70).sum()
                        oversold = (rsi_df['rsi'] < 30).sum()
                        
                        f.write(f"#### {symbol}\n\n")
                        f.write(f"- Overbought Periods (RSI > 70): {overbought}\n")
                        f.write(f"- Oversold Periods (RSI < 30): {oversold}\n\n")
                        f.write(f"![{symbol} RSI]({symbol}_rsi.png)\n\n")
                
                # Correlation summary
                if os.path.exists(self.results_dir / "correlation_matrix.csv"):
                    f.write("### Correlation Analysis\n\n")
                    f.write("![Correlation Heatmap](correlation_heatmap.png)\n\n")
                
                f.write("## Conclusion\n\n")
                f.write("This report provides a comprehensive analysis of the selected stocks. ")
                f.write("For detailed data, please refer to the individual CSV files and charts in the analysis_results directory.\n")
            
            print(f"Summary report generated at {report_path}")
            return report_path
            
        except Exception as e:
            print(f"Error generating summary report: {e}")
            return None

def main():
    # Initialize analyzer
    analyzer = StockAnalyzer()
    
    # Define symbols to analyze
    symbols = input("Enter stock symbols to analyze (comma-separated, e.g., AAPL,MSFT,GOOGL): ").split(',')
    symbols = [symbol.strip().upper() for symbol in symbols]
    
    if not symbols:
        print("No symbols provided. Using default symbols.")
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Get data
    print("\nFetching historical stock data...")
    stock_data = analyzer.get_stock_data(symbols, interval="day", span="year")
    
    print("\nFetching fundamentals...")
    fundamentals = analyzer.get_fundamentals(symbols)
    
    # Analyze data
    print("\nAnalyzing volatility...")
    volatility_results = analyzer.analyze_volatility(stock_data)
    print(volatility_results)
    
    print("\nAnalyzing moving averages...")
    ma_signals = analyzer.analyze_moving_averages(stock_data)
    
    print("\nAnalyzing RSI...")
    rsi_results = analyzer.analyze_rsi(stock_data)
    
    print("\nAnalyzing correlation...")
    correlation_matrix = analyzer.analyze_correlation(stock_data)
    if correlation_matrix is not None:
        print(correlation_matrix)
    
    # Generate summary report
    print("\nGenerating summary report...")
    report_path = analyzer.generate_summary_report(symbols)
    
    print("\nAnalysis complete! Results are saved in the 'analysis_results' directory.")

if __name__ == "__main__":
    main()
