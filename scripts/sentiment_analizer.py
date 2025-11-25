import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
from collections import Counter
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class FinancialSentimentAnalyzer:
   
    
    def __init__(self, file_path=None):
       
        if file_path:
            self.data = self.load_data(file_path)
        
        else:
            raise ValueError("file_path  must be provided")
        
        self.clean_data()
        self.prepare_features()
        logger.info(f"Analyzer initialized with {len(self.data)} records")
    
    def load_data(self, file_path):
        """Load and initial data processing"""
        logger.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path, index_col=0)
        return data
    
    def clean_data(self):
        """Clean and preprocess the data"""
        logger.info("Cleaning data...")
        
        # Remove null headlines
        initial_count = len(self.data)
        self.data.dropna(subset=['headline'], inplace=True)
        self.data['headline'] = self.data['headline'].astype(str)
        
        # Remove duplicates
        self.data.drop_duplicates(inplace=True)
        
        # Convert dates
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'], format='ISO8601', utc=True)
        
        logger.info(f"Data cleaning complete. Removed {initial_count - len(self.data)} records")
    
    def prepare_features(self):
        """Create additional features for analysis"""
        logger.info("Creating features...")
        
        # Text features
        self.data['headline_length'] = self.data['headline'].apply(len)
        self.data['word_count'] = self.data['headline'].apply(lambda x: len(x.split()))
        
        # Date features
        if 'date' in self.data.columns:
            self.data['date_only'] = self.data['date'].dt.date
            self.data['hour'] = self.data['date'].dt.hour
            self.data['day_of_week'] = self.data['date'].dt.day_name()
            self.data['month'] = self.data['date'].dt.to_period('M')
            self.data['week'] = self.data['date'].dt.isocalendar().week
    
    def descriptive_statistics(self):
        """Generate comprehensive descriptive statistics"""
        logger.info("Generating descriptive statistics...")
        
        stats = {
            'total_articles': len(self.data),
            'unique_publishers': self.data['publisher'].nunique(),
            'date_range': f"{self.data['date'].min()} to {self.data['date'].max()}",
            'headline_length_mean': self.data['headline_length'].mean(),
            'headline_length_std': self.data['headline_length'].std(),
            'word_count_mean': self.data['word_count'].mean(),
            'word_count_std': self.data['word_count'].std()
        }
        
        return stats
    
    def analyze_publishers(self, top_n=15):
        """Analyze publisher activity and influence"""
        logger.info("Analyzing publisher activity...")
        
        publisher_counts = self.data['publisher'].value_counts().head(top_n)
        publisher_stats = {
            'top_publishers': publisher_counts.to_dict(),
            'total_unique': self.data['publisher'].nunique(),
            'top_5_coverage': publisher_counts.head(5).sum() / len(self.data) * 100
        }
        
        return publisher_stats
    
    def temporal_analysis(self):
        """Analyze publication patterns over time"""
        logger.info("Performing temporal analysis...")
        
        if 'date' not in self.data.columns:
            return {"error": "Date column not available"}
        
        # Daily counts
        daily_counts = self.data.groupby('date_only').size()
        
        # Hourly distribution
        hourly_counts = self.data['hour'].value_counts().sort_index()
        
        # Day of week distribution
        dow_counts = self.data['day_of_week'].value_counts()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_counts = dow_counts.reindex(dow_order)
        
        temporal_stats = {
            'peak_hour': hourly_counts.idxmax(),
            'peak_hour_count': hourly_counts.max(),
            'most_active_day': dow_counts.idxmax(),
            'most_active_day_count': dow_counts.max(),
            'avg_daily_articles': daily_counts.mean(),
            'total_days': daily_counts.nunique()
        }
        
        return temporal_stats
    
    def extract_keywords(self, top_n=20):
        """Extract and analyze common keywords"""
        logger.info("Extracting keywords...")
        
        def extract_keywords_text(text):
            text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text))
            words = text.lower().split()
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                         'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been'}
            keywords = [word for word in words if word not in stop_words and len(word) > 2]
            return keywords
        
        all_keywords = []
        for headline in self.data['headline']:
            all_keywords.extend(extract_keywords_text(headline))
        
        keyword_freq = Counter(all_keywords)
        common_keywords = dict(keyword_freq.most_common(top_n))
        
        return common_keywords
    
    def analyze_topics(self):
        """Analyze financial topics in headlines"""
        logger.info("Analyzing topics...")
        
        key_topics = {
            'earnings': ['earnings', 'profit', 'revenue', 'quarter', 'results', 'eps'],
            'price_targets': ['target', 'price', 'raise', 'lower', 'maintain', 'pt'],
            'analyst_ratings': ['upgrade', 'downgrade', 'initiate', 'coverage', 'rating', 'analyst'],
            'market_movements': ['stock', 'shares', 'trading', 'market', 'high', 'low', 'gain', 'drop'],
            'corporate_actions': ['merger', 'acquisition', 'dividend', 'split', 'buyback', 'offer'],
            'regulatory': ['fda', 'approval', 'regulation', 'investigation', 'lawsuit'],
            'guidance': ['forecast', 'outlook', 'guidance', 'expect', 'projection']
        }
        
        topic_counts = {topic: 0 for topic in key_topics.keys()}
        topic_examples = {topic: [] for topic in key_topics.keys()}
        
        for idx, row in self.data.iterrows():
            headline_lower = row['headline'].lower()
            for topic, keywords in key_topics.items():
                if any(keyword in headline_lower for keyword in keywords):
                    topic_counts[topic] += 1
                    if len(topic_examples[topic]) < 3:  # Keep 3 examples per topic
                        topic_examples[topic].append(row['headline'])
        
        return {
            'topic_counts': topic_counts,
            'topic_examples': topic_examples
        }
    
    def perform_sentiment_analysis(self, sample_size=5000):
        """Perform comprehensive sentiment analysis"""
        logger.info("Performing sentiment analysis...")
        
        # Sample if dataset is large
        if len(self.data) > sample_size:
            sentiment_sample = self.data.sample(sample_size, random_state=42)
            logger.info(f"Using sample of {sample_size} records for sentiment analysis")
        else:
            sentiment_sample = self.data
        
        def get_sentiment(text):
            analysis = TextBlob(str(text))
            return analysis.sentiment.polarity
        
        sentiment_sample = sentiment_sample.copy()
        sentiment_sample['sentiment'] = sentiment_sample['headline'].apply(get_sentiment)
        
        def sentiment_category(score):
            if score >= 0.1:
                return 'Positive'
            elif score <= -0.1:
                return 'Negative'
            else:
                return 'Neutral'
        
        sentiment_sample['sentiment_category'] = sentiment_sample['sentiment'].apply(sentiment_category)
        
        sentiment_counts = sentiment_sample['sentiment_category'].value_counts()
        
        # Sentiment by publisher
        top_publishers = self.data['publisher'].value_counts().head(5).index
        publisher_sentiment = sentiment_sample[sentiment_sample['publisher'].isin(top_publishers)]
        publisher_sentiment_stats = publisher_sentiment.groupby('publisher')['sentiment'].agg(['mean', 'std', 'count']).to_dict('index')
        
        # Temporal sentiment
        if 'date' in sentiment_sample.columns:
            monthly_sentiment = sentiment_sample.groupby('month')['sentiment'].mean()
            hourly_sentiment = sentiment_sample.groupby('hour')['sentiment'].mean()
        else:
            monthly_sentiment = None
            hourly_sentiment = None
        
        # Extreme sentiments
        most_positive = sentiment_sample.loc[sentiment_sample['sentiment'].idxmax()]
        most_negative = sentiment_sample.loc[sentiment_sample['sentiment'].idxmin()]
        
        sentiment_results = {
            'overall_distribution': sentiment_counts.to_dict(),
            'average_sentiment': sentiment_sample['sentiment'].mean(),
            'sentiment_std': sentiment_sample['sentiment'].std(),
            'publisher_sentiment': publisher_sentiment_stats,
            'most_positive': {
                'headline': most_positive['headline'],
                'sentiment': most_positive['sentiment'],
                'publisher': most_positive['publisher']
            },
            'most_negative': {
                'headline': most_negative['headline'],
                'sentiment': most_negative['sentiment'],
                'publisher': most_negative['publisher']
            },
            'sample_size': len(sentiment_sample)
        }
        
        if monthly_sentiment is not None:
            sentiment_results['monthly_sentiment'] = monthly_sentiment.to_dict()
            sentiment_results['hourly_sentiment'] = hourly_sentiment.to_dict()
        
        return sentiment_results, sentiment_sample
    
    def generate_visualizations(self, sentiment_sample=None, save_path=None):
        """Generate comprehensive visualizations"""
        logger.info("Generating visualizations...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Publisher activity
        plt.subplot(3, 3, 1)
        publisher_counts = self.data['publisher'].value_counts().head(10)
        publisher_counts.plot(kind='bar', color='skyblue')
        plt.title('Top 10 Most Active Publishers')
        plt.xlabel('Publisher')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        
        # 2. Temporal patterns
        if 'date' in self.data.columns:
            # Hourly distribution
            plt.subplot(3, 3, 2)
            hourly_counts = self.data['hour'].value_counts().sort_index()
            hourly_counts.plot(kind='bar', color='lightcoral')
            plt.title('Publication by Hour of Day')
            plt.xlabel('Hour (24h)')
            plt.ylabel('Number of Articles')
            
            # Day of week distribution
            plt.subplot(3, 3, 3)
            dow_counts = self.data['day_of_week'].value_counts()
            dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_counts = dow_counts.reindex(dow_order)
            dow_counts.plot(kind='bar', color='lightgreen')
            plt.title('Publication by Day of Week')
            plt.xlabel('Day of Week')
            plt.ylabel('Number of Articles')
            plt.xticks(rotation=45)
        
        # 3. Topic analysis
        plt.subplot(3, 3, 4)
        topic_results = self.analyze_topics()
        topics = list(topic_results['topic_counts'].keys())
        counts = list(topic_results['topic_counts'].values())
        plt.barh(topics, counts, color='orange')
        plt.title('Frequency of Key Topics')
        plt.xlabel('Number of Articles')
        
        # 4. Keyword analysis
        plt.subplot(3, 3, 5)
        keywords = self.extract_keywords(10)
        plt.barh(list(keywords.keys())[:10], list(keywords.values())[:10], color='purple')
        plt.title('Top 10 Keywords')
        plt.xlabel('Frequency')
        
        # 5. Sentiment analysis
        if sentiment_sample is not None:
            plt.subplot(3, 3, 6)
            sentiment_counts = sentiment_sample['sentiment_category'].value_counts()
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
            plt.title('Sentiment Distribution')
            
            # 6. Sentiment by publisher
            plt.subplot(3, 3, 7)
            top_publishers = self.data['publisher'].value_counts().head(5).index
            publisher_sentiment_data = sentiment_sample[sentiment_sample['publisher'].isin(top_publishers)]
            sns.boxplot(data=publisher_sentiment_data, x='publisher', y='sentiment')
            plt.title('Sentiment by Top Publishers')
            plt.xticks(rotation=45)
            
            # 7. Sentiment over time
            if 'month' in sentiment_sample.columns:
                plt.subplot(3, 3, 8)
                monthly_sentiment = sentiment_sample.groupby('month')['sentiment'].mean()
                monthly_sentiment.plot()
                plt.title('Average Sentiment Over Time')
                plt.xlabel('Month')
                plt.ylabel('Average Sentiment')
                plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualizations saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, save_visualizations=None):
        """Generate comprehensive analysis report"""
        logger.info("Generating comprehensive report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': self.descriptive_statistics(),
            'publisher_analysis': self.analyze_publishers(),
            'temporal_analysis': self.temporal_analysis(),
            'keyword_analysis': self.extract_keywords(20),
            'topic_analysis': self.analyze_topics()
        }
        
        # Perform sentiment analysis
        sentiment_results, sentiment_sample = self.perform_sentiment_analysis()
        report['sentiment_analysis'] = sentiment_results
        
        # Generate insights
        report['key_insights'] = self._generate_insights(report)
        
        # Generate visualizations
        if save_visualizations:
            self.generate_visualizations(sentiment_sample, save_visualizations)
        else:
            self.generate_visualizations(sentiment_sample)
        
        return report
    
    def _generate_insights(self, report):
        """Generate key insights from analysis results"""
        insights = []
        
        # Dataset insights
        insights.append(f" Dataset contains {report['dataset_info']['total_articles']:,} articles from {report['dataset_info']['unique_publishers']} publishers")
        
        # Publisher insights
        top_pub = list(report['publisher_analysis']['top_publishers'].keys())[0]
        insights.append(f" Top publisher '{top_pub}' contributed {report['publisher_analysis']['top_publishers'][top_pub]:,} articles")
        
        # Temporal insights
        if 'peak_hour' in report['temporal_analysis']:
            insights.append(f" Peak publishing hour: {report['temporal_analysis']['peak_hour']}:00")
            insights.append(f" Most active day: {report['temporal_analysis']['most_active_day']}")
        
        # Sentiment insights
        sentiment = report['sentiment_analysis']
        pos_pct = (sentiment['overall_distribution'].get('Positive', 0) / sentiment['sample_size']) * 100
        neg_pct = (sentiment['overall_distribution'].get('Negative', 0) / sentiment['sample_size']) * 100
        insights.append(f" Sentiment: {pos_pct:.1f}% Positive, {neg_pct:.1f}% Negative")
        insights.append(f" Average sentiment score: {sentiment['average_sentiment']:.3f}")
        
        # Topic insights
        topics = report['topic_analysis']['topic_counts']
        dominant_topic = max(topics, key=topics.get)
        insights.append(f" Dominant topic: {dominant_topic.replace('_', ' ').title()} ({topics[dominant_topic]} mentions)")
        
        return insights
    
