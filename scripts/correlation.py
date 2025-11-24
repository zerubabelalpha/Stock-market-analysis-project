import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from datetime import datetime, timedelta

class CorrelationAnalysis:
    """Handles correlation analysis between news sentiment and stock movements"""
    
    def __init__(self):
        self.merged_data = None
        self.news_data = None
        self.stock_data = None
    
    def load_and_prepare_data(self, news_data_path, stock_data_path):
        """
        Load and prepare news and stock data for correlation analysis
        """
        # Load and preprocess news data
        print("Loading news data...")
        news_data = pd.read_csv(news_data_path, index_col=0)
        
        # Clean news data
        news_data.drop_duplicates(inplace=True)
        news_data.dropna(subset=['headline'], inplace=True)
        
        # Convert date and set index - REMOVE TIMEZONE for alignment
        news_data['date'] = pd.to_datetime(news_data['date'], format='ISO8601', utc=True)
        news_data['date'] = news_data['date'].dt.tz_convert(None)  # Remove timezone
        news_data.set_index('date', inplace=True)
        
        # Calculate sentiment
        news_data['sentiment'] = news_data['headline'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity
        )
        
        # Load and preprocess stock data
        print("Loading stock data...")
        stock_data = pd.read_csv(stock_data_path)
        
        # Ensure Date column is parsed correctly and set as index
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data.set_index('Date', inplace=True)
        
        # Calculate daily returns
        stock_data['daily_return'] = stock_data['Close'].pct_change() * 100
        
        # Aggregate news sentiment by day
        print("Aggregating data by date...")
        daily_sentiment = news_data['sentiment'].resample('D').agg(['mean', 'count'])
        daily_sentiment.columns = ['avg_sentiment', 'news_count']
        
        # Prepare stock returns by day
        daily_returns = stock_data['daily_return'].resample('D').last()
        
        # Merge datasets on date (both should now be timezone-naive)
        merged_data = pd.concat([daily_sentiment, daily_returns], axis=1, join='inner')
        merged_data.dropna(inplace=True)
        
        self.merged_data = merged_data
        self.news_data = news_data
        self.stock_data = stock_data
        
        print(f" Data preparation completed!")
        print(f"Analysis period: {merged_data.index.min()} to {merged_data.index.max()}")
        print(f"Total days with both news and stock data: {len(merged_data)}")
        
        return merged_data, news_data, stock_data

    def calculate_correlations(self):
        """
        Calculate various correlation metrics
        """
        if self.merged_data is None:
            print(" Please load data first using load_and_prepare_data()")
            return None, None
        
        print("\n=== CORRELATION ANALYSIS ===")
        
        # Basic Pearson correlation
        corr_coef, p_value = pearsonr(
            self.merged_data['avg_sentiment'], 
            self.merged_data['daily_return']
        )
        
        print(f"Pearson Correlation: {corr_coef:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(" Correlation is statistically significant (p < 0.05)")
        else:
            print(" Correlation is not statistically significant")
        
        # Interpret correlation strength
        abs_corr = abs(corr_coef)
        if abs_corr >= 0.7:
            strength = "strong"
        elif abs_corr >= 0.5:
            strength = "moderate" 
        elif abs_corr >= 0.3:
            strength = "weak"
        else:
            strength = "very weak"
        
        print(f"Correlation strength: {strength}")
        
        # Lagged correlations (to account for potential delayed effects)
        print("\n=== LAGGED CORRELATIONS ===")
        max_lags = min(5, len(self.merged_data) // 10)
        for lag in range(1, max_lags + 1):
            sentiment_lagged = self.merged_data['avg_sentiment'].shift(lag)
            valid_data = pd.concat([sentiment_lagged, self.merged_data['daily_return']], axis=1).dropna()
            
            if len(valid_data) > 2:
                corr_lag, p_lag = pearsonr(
                    valid_data['avg_sentiment'], 
                    valid_data['daily_return']
                )
                significance = "✅" if p_lag < 0.05 else "_X_"
                print(f"Lag {lag} day(s): {corr_lag:.4f} (p-value: {p_lag:.4f}) {significance}")
        
        return corr_coef, p_value

    def create_correlation_visualizations(self):
        """
        Create visualizations for the correlation analysis
        """
        if self.merged_data is None:
            print(" Please load data first using load_and_prepare_data()")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Sentiment vs Returns Scatter
        axes[0, 0].scatter(self.merged_data['avg_sentiment'], self.merged_data['daily_return'], alpha=0.6)
        axes[0, 0].set_xlabel('Average Daily Sentiment')
        axes[0, 0].set_ylabel('Daily Return (%)')
        axes[0, 0].set_title('Sentiment vs Stock Returns')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add trend line and correlation info
        z = np.polyfit(self.merged_data['avg_sentiment'], self.merged_data['daily_return'], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(self.merged_data['avg_sentiment'], p(self.merged_data['avg_sentiment']), "r--", alpha=0.8)
        
        corr_coef, p_value = pearsonr(self.merged_data['avg_sentiment'], self.merged_data['daily_return'])
        axes[0, 0].text(0.05, 0.95, f'Correlation: {corr_coef:.3f}\np-value: {p_value:.3f}', 
                       transform=axes[0, 0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: Time series of sentiment and returns
        ax2 = axes[0, 1]
        ax2.plot(self.merged_data.index, self.merged_data['avg_sentiment'], label='Avg Sentiment', color='blue', alpha=0.7)
        ax2.set_ylabel('Sentiment Score', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylim([-1, 1])
        
        ax2b = ax2.twinx()
        ax2b.plot(self.merged_data.index, self.merged_data['daily_return'], label='Daily Return', color='red', alpha=0.7)
        ax2b.set_ylabel('Daily Return (%)', color='red')
        ax2b.tick_params(axis='y', labelcolor='red')
        
        ax2.set_title('Sentiment and Returns Over Time')
        
        # Plot 3: News volume vs returns
        axes[1, 0].scatter(self.merged_data['news_count'], self.merged_data['daily_return'], alpha=0.6)
        axes[1, 0].set_xlabel('Number of News Articles')
        axes[1, 0].set_ylabel('Daily Return (%)')
        axes[1, 0].set_title('News Volume vs Stock Returns')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Sentiment distribution by return direction
        self.merged_data['return_direction'] = np.where(
            self.merged_data['daily_return'] > 0, 'Positive', 'Negative'
        )
        
        sns.boxplot(data=self.merged_data, x='return_direction', y='avg_sentiment', ax=axes[1, 1])
        axes[1, 1].set_title('Sentiment Distribution by Return Direction')
        axes[1, 1].set_xlabel('Daily Return Direction')
        axes[1, 1].set_ylabel('Average Sentiment')
        
        plt.tight_layout()
        plt.show()

    def advanced_correlation_analysis(self):
        """
        Additional correlation analysis methods
        """
        if self.merged_data is None:
            print(" Please load data first using load_and_prepare_data()")
            return
        
        print("\n=== ADVANCED ANALYSIS ===")
        
        # Correlation by sentiment intensity
        self.merged_data['sentiment_abs'] = self.merged_data['avg_sentiment'].abs()
        high_sentiment_days = self.merged_data[self.merged_data['sentiment_abs'] > self.merged_data['sentiment_abs'].median()]
        
        if len(high_sentiment_days) > 2:
            corr_high, p_high = pearsonr(
                high_sentiment_days['avg_sentiment'], 
                high_sentiment_days['daily_return']
            )
            print(f"Correlation on high-sentiment days: {corr_high:.4f} (p-value: {p_high:.4f})")
        
        # Rolling correlation (30-day window)
        if len(self.merged_data) > 30:
            rolling_corr = self.merged_data['avg_sentiment'].rolling(window=30).corr(self.merged_data['daily_return'])
            
            plt.figure(figsize=(12, 6))
            plt.plot(rolling_corr.index, rolling_corr.values)
            plt.title('30-Day Rolling Correlation: Sentiment vs Returns')
            plt.xlabel('Date')
            plt.ylabel('Correlation Coefficient')
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            plt.grid(True, alpha=0.3)
            plt.show()
            
            print(f"Average rolling correlation: {rolling_corr.mean():.4f}")
        else:
            print("Not enough data for rolling correlation analysis (need >30 days)")
    
    def sentiment_category_analysis(self):
        """
        Analyze returns by sentiment category
        """
        if self.merged_data is None:
            print(" Please load data first using load_and_prepare_data()")
            return
        
        print("\n=== SENTIMENT CATEGORY ANALYSIS ===")
        
        # Categorize sentiment
        def get_sentiment_category(score):
            if score > 0.1:
                return 'Positive'
            elif score < -0.1:
                return 'Negative'
            else:
                return 'Neutral'
        
        self.merged_data['sentiment_category'] = self.merged_data['avg_sentiment'].apply(get_sentiment_category)
        
        # Calculate average returns by category
        category_returns = self.merged_data.groupby('sentiment_category')['daily_return'].agg(['mean', 'std', 'count'])
        print("Average Returns by Sentiment Category:")
        print(category_returns)
        
        # Plot category returns
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.merged_data, x='sentiment_category', y='daily_return', 
                    order=['Negative', 'Neutral', 'Positive'])
        plt.title('Stock Returns by Sentiment Category')
        plt.xlabel('Sentiment Category')
        plt.ylabel('Daily Return (%)')
        plt.grid(True, alpha=0.3)
        plt.show()

    def get_summary_statistics(self):
        """
        Get summary statistics of the merged data
        """
        if self.merged_data is None:
            print(" Please load data first using load_and_prepare_data()")
            return
        
        print("\n=== SUMMARY STATISTICS ===")
        return self.merged_data[['avg_sentiment', 'daily_return', 'news_count']].describe()





# Example usage
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = CorrelationAnalysis()
    
    # Load your data
    news_data_path = "../data/newsData/raw_analyst_ratings.csv"
    stock_data_path = "../data/yfinance_data/Data/AAPL.csv"
    
    try:
        # Load and prepare data
        merged_data, news_data, stock_data = analyzer.load_and_prepare_data(news_data_path, stock_data_path)
        
        if len(merged_data) > 0:
            # Run all analyses
            analyzer.calculate_correlations()
            analyzer.create_correlation_visualizations()
            analyzer.advanced_correlation_analysis()
            analyzer.sentiment_category_analysis()
            
            # Print summary statistics
            print(analyzer.get_summary_statistics())
        else:
            print(" No overlapping data found between news and stock datasets!")
            
    except Exception as e:
        print(f" Error during analysis: {e}")
        import traceback
        print(traceback.format_exc())