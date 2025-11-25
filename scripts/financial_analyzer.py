import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from datetime import datetime
import talib

class FinancialDataLoader:
   
    
    def __init__(self):
        self.data = None
        
    def load_from_csv(self, file_path, date_col='Date', index_col=None):
        
        try:
            self.data = pd.read_csv(file_path, parse_dates=[date_col])
            if index_col:
                self.data.set_index(index_col, inplace=True)
            elif date_col:
                self.data.set_index(date_col, inplace=True)
                
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def validate_data(self, required_cols=['Open', 'High', 'Low', 'Close', 'Volume']):
        """
        Validate that required columns exist in the data
        
        Args:
            required_cols (list): List of required column names
        """
        if self.data is None:
            print("No data loaded")
            return False
        
        # Handle MultiIndex columns - flatten them
        if isinstance(self.data.columns, pd.MultiIndex):
            print("Flattening MultiIndex columns...")
            self.data.columns = self.data.columns.get_level_values(0)  # Keep only first level
        
        # Ensure index is datetime and sorted
        if not isinstance(self.data.index, pd.DatetimeIndex):
            print("Converting index to datetime...")
            try:
                self.data.index = pd.to_datetime(self.data.index)
            except Exception as e:
                print(f"Error converting index to datetime: {e}")
                return False
        
        # Sort by datetime index
        self.data = self.data.sort_index()
        
        # Check for required columns
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            print(f"Available columns: {self.data.columns.tolist()}")
            return False
            
        # Clean data - keep only required columns and drop NaN
        self.data = self.data[required_cols].dropna()
        print(f"Data validated. Final shape: {self.data.shape}")
        print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
        return True
    
    def get_data(self):
        """Return the loaded data"""
        return self.data
    
    def add_ticker_column(self, ticker):
        """Add a ticker column to the data (useful for multi-ticker analysis)"""
        if self.data is not None:
            self.data['Ticker'] = ticker
            print(f"Added ticker column: {ticker}")

class TechnicalAnalyzer:
    """Handles technical analysis using TA-Lib"""
    
    def __init__(self, data):
        self.data = data
        self.indicators = {}
        
    def calculate_moving_averages(self):
        """Calculate various moving averages"""
        price = self.data['Close']
        
        self.data['SMA_20'] = talib.SMA(price, timeperiod=20)
        self.data['SMA_50'] = talib.SMA(price, timeperiod=50)
        self.data['EMA_12'] = talib.EMA(price, timeperiod=12)
        self.data['EMA_26'] = talib.EMA(price, timeperiod=26)
        
        print("Moving averages calculated")
        
    def calculate_momentum_indicators(self):
        """Calculate momentum indicators"""
        price = self.data['Close']
        
        # RSI
        self.data['RSI_14'] = talib.RSI(price, timeperiod=14)
        
        # MACD
        self.data['MACD'], self.data['MACD_signal'], self.data['MACD_hist'] = talib.MACD(price)
        
        # Stochastic
        self.data['STOCH_K'], self.data['STOCH_D'] = talib.STOCH(
            self.data['High'], self.data['Low'], self.data['Close']
        )
        
        print("Momentum indicators calculated")
        
    def calculate_volatility_indicators(self):
        """Calculate volatility indicators"""
        price = self.data['Close']
        
        # Bollinger Bands
        self.data['BB_upper'], self.data['BB_middle'], self.data['BB_lower'] = talib.BBANDS(price)
        
        # ATR (Average True Range)
        self.data['ATR_14'] = talib.ATR(
            self.data['High'], self.data['Low'], self.data['Close'], timeperiod=14
        )
        
        print("Volatility indicators calculated")
        
    def calculate_all_indicators(self):
        """Calculate all technical indicators"""
        self.calculate_moving_averages()
        self.calculate_momentum_indicators()
        self.calculate_volatility_indicators()
        
        # Calculate returns
        self.data['Daily_Return'] = self.data['Close'].pct_change()
        self.data['Cumulative_Return'] = (1 + self.data['Daily_Return']).cumprod() - 1
        
        return self.data
    
    def get_indicators_summary(self):
        """Get statistical summary of calculated indicators"""
        indicator_cols = [col for col in self.data.columns if col not in 
                         ['Open', 'High', 'Low', 'Close', 'Volume']]
        return self.data[indicator_cols].describe()


class PortfolioAnalyzer:
    """Handles portfolio analysis using PyNance"""
    
    def __init__(self, tickers):
        self.tickers = tickers
        self.portfolio = None
        
    def initialize_portfolio(self):
        """Initialize portfolio calculator"""
        try:
            from pynance import portfolio_optimizer as po
            self.portfolio = po.PortfolioCalculations(self.tickers)
            print("Portfolio analyzer initialized")
            return True
        except Exception as e:
            print(f"Error initializing portfolio analyzer: {e}")
            return False
    
    def get_max_sharpe_portfolio(self):
        """Get maximum Sharpe ratio portfolio"""
        if self.portfolio:
            try:
                return self.portfolio.max_sharpe_portfolio("rr")
            except Exception as e:
                print(f"Error calculating max Sharpe portfolio: {e}")
                return None
        return None
    
    def get_min_variance_portfolio(self):
        """Get minimum variance portfolio"""
        if self.portfolio:
            try:
                return self.portfolio.min_var_portfolio("rr")
            except Exception as e:
                print(f"Error calculating min variance portfolio: {e}")
                return None
        return None
    
    def get_portfolio_weights(self, portfolio_type='max_sharpe'):
        """Get portfolio weights"""
        if self.portfolio:
            try:
                if portfolio_type == 'max_sharpe':
                    return self.portfolio.max_sharpe_portfolio("df")
                else:
                    return self.portfolio.min_var_portfolio("df")
            except Exception as e:
                print(f"Error getting portfolio weights: {e}")
                return None
        return None


class FinancialVisualizer:
    """Handles visualization of financial data and indicators"""
    
    def __init__(self, data):
        self.data = data
        self.setup_plotting()
        
    def setup_plotting(self):
        """Setup matplotlib parameters"""
        plt.style.use("seaborn-v0_8")
        plt.rcParams["figure.figsize"] = (12, 6)
        
    def plot_price_indicators(self):
        """Plot price with technical indicators"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Price with Moving Averages
        axes[0].plot(self.data.index, self.data['Close'], 
                    label='Close Price', color='black', linewidth=1)
        if 'SMA_20' in self.data.columns:
            axes[0].plot(self.data.index, self.data['SMA_20'], 
                        label='SMA 20', linestyle='--', alpha=0.8)
        if 'SMA_50' in self.data.columns:
            axes[0].plot(self.data.index, self.data['SMA_50'], 
                        label='SMA 50', linestyle='-.', alpha=0.8)
        axes[0].set_title('Price with Moving Averages')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RSI
        if 'RSI_14' in self.data.columns:
            axes[1].plot(self.data.index, self.data['RSI_14'], 
                        label='RSI 14', color='purple')
            axes[1].axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought')
            axes[1].axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold')
            axes[1].set_title('RSI - Relative Strength Index')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # MACD
        if 'MACD' in self.data.columns:
            axes[2].plot(self.data.index, self.data['MACD'], 
                        label='MACD', color='blue')
            axes[2].plot(self.data.index, self.data['MACD_signal'], 
                        label='Signal', color='red')
            axes[2].bar(self.data.index, self.data['MACD_hist'], 
                       label='Histogram', alpha=0.3)
            axes[2].set_title('MACD')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_volume_analysis(self):
        """Plot volume analysis"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Price
        ax1.plot(self.data.index, self.data['Close'], color='black', linewidth=1)
        ax1.set_title('Close Price')
        ax1.grid(True, alpha=0.3)
        
        # Volume (colored by price movement)
        colors = ['red' if self.data['Close'].iloc[i] < self.data['Open'].iloc[i] 
                 else 'green' for i in range(len(self.data))]
        ax2.bar(self.data.index, self.data['Volume'], color=colors, alpha=0.6)
        ax2.set_title('Daily Volume (Red=Down, Green=Up)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_returns_analysis(self):
        """Plot returns distribution and cumulative returns"""
        if 'Daily_Return' not in self.data.columns:
            print("Daily returns not calculated")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Returns distribution
        ax1.hist(self.data['Daily_Return'].dropna(), bins=50, 
                alpha=0.7, edgecolor='black')
        ax1.set_title('Distribution of Daily Returns')
        ax1.set_xlabel('Daily Return')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative returns
        if 'Cumulative_Return' in self.data.columns:
            ax2.plot(self.data.index, self.data['Cumulative_Return'] * 100)
        else:
            cumulative_returns = (1 + self.data['Daily_Return']).cumprod() - 1
            ax2.plot(self.data.index, cumulative_returns * 100)
            
        ax2.set_title('Cumulative Returns (%)')
        ax2.set_ylabel('Cumulative Return (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap of indicators"""
        try:
            # Select numeric columns for correlation
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            correlation_data = self.data[numeric_cols].corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_data, annot=True, cmap='coolwarm', 
                       center=0, fmt='.2f', linewidths=0.5)
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error creating correlation heatmap: {e}")


class FinancialAnalysis:
    """Main class to coordinate financial analysis"""
    
    def __init__(self):
        self.data_loader = FinancialDataLoader()
        self.technical_analyzer = None
        self.visualizer = None
        self.portfolio_analyzer = None
        
    def load_and_prepare_data(self, file_path, **kwargs):
        """Load and prepare data for analysis"""
        data = self.data_loader.load_from_csv(file_path, **kwargs)
        if data is not None:
            if self.data_loader.validate_data():
                self.technical_analyzer = TechnicalAnalyzer(self.data_loader.get_data())
                self.visualizer = FinancialVisualizer(self.data_loader.get_data())
                return True
        return False
    
    def run_technical_analysis(self):
        """Run complete technical analysis"""
        if self.technical_analyzer:
            data_with_indicators = self.technical_analyzer.calculate_all_indicators()
            summary = self.technical_analyzer.get_indicators_summary()
            print("Technical Analysis Summary:")
            print(summary)
            return data_with_indicators
        else:
            print("Technical analyzer not initialized")
            return None
    
    def visualize_results(self):
        """Create all visualizations"""
        if self.visualizer:
            self.visualizer.plot_price_indicators()
            self.visualizer.plot_volume_analysis()
            self.visualizer.plot_returns_analysis()
            self.visualizer.plot_correlation_heatmap()
        else:
            print("Visualizer not initialized")
    
    def run_portfolio_analysis(self, tickers):
        """Run portfolio analysis"""
        self.portfolio_analyzer = PortfolioAnalyzer(tickers)
        if self.portfolio_analyzer.initialize_portfolio():
            print("\n=== Portfolio Analysis Results ===")
            
            max_sharpe = self.portfolio_analyzer.get_max_sharpe_portfolio()
            if max_sharpe is not None:
                print("Max Sharpe Portfolio:")
                print(max_sharpe)
            
            min_var = self.portfolio_analyzer.get_min_variance_portfolio()
            if min_var is not None:
                print("\nMin Variance Portfolio:")
                print(min_var)
                
            return True
        return False


