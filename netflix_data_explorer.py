#!/usr/bin/env python3
"""
Netflix Data Explorer Challenge
A comprehensive analysis of Netflix titles using BERT sentiment analysis and interactive visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NetflixDataExplorer:
    def __init__(self, csv_path='netflix_titles.csv'):
        """Initialize the Netflix Data Explorer."""
        self.csv_path = csv_path
        self.df = None
        self.sentiment_pipeline = None
        
    def load_data(self):
        """Task 1.1: Load the netflix_titles.csv file into a Pandas DataFrame."""
        try:
            self.df = pd.read_csv(self.csv_path)
            print("✅ Data loaded successfully!")
            return True
        except FileNotFoundError:
            print(f"❌ Error: File '{self.csv_path}' not found.")
            print("Please ensure the netflix_titles.csv file is in the current directory.")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def display_basic_info(self):
        """Task 1.2: Display basic information about the DataFrame."""
        print("\n" + "="*60)
        print("📊 BASIC DATAFRAME INFORMATION")
        print("="*60)
        
        print(f"DataFrame Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Data Types:\n{self.df.dtypes}")
        
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nLast 5 rows:")
        print(self.df.tail())
    
    def analyze_missing_values(self):
        """Task 1.3: Identify and print the number of missing values per column."""
        print("\n" + "="*60)
        print("🔍 MISSING VALUES ANALYSIS")
        print("="*60)
        
        # Calculate missing values for each column
        missing_values = self.df.isnull().sum()
        missing_percentage = (missing_values / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_values.index,
            'Missing_Count': missing_values.values,
            'Missing_Percentage': missing_percentage.values
        })
        
        print("Missing values per column:")
        print(missing_df.to_string(index=False))
        
        return missing_df
    
    def fill_missing_values(self):
        """Task 1.4: Fill missing values accordingly."""
        print("\n" + "="*60)
        print("🧹 FILLING MISSING VALUES")
        print("="*60)
        
        # Store original missing count for comparison
        original_missing = self.df.isnull().sum().sum()
        
        # Fill missing values based on data type and context
        self.df['director'].fillna('Unknown Director', inplace=True)
        self.df['cast'].fillna('Unknown Cast', inplace=True)
        self.df['country'].fillna('Unknown Country', inplace=True)
        self.df['date_added'].fillna('Unknown Date', inplace=True)
        self.df['rating'].fillna('Unknown Rating', inplace=True)
        self.df['duration'].fillna('Unknown Duration', inplace=True)
        self.df['listed_in'].fillna('Unknown Genre', inplace=True)
        self.df['description'].fillna('No description available', inplace=True)
        
        # For numeric columns, fill with appropriate values
        self.df['release_year'].fillna(self.df['release_year'].median(), inplace=True)
        
        # Verify missing values are filled
        final_missing = self.df.isnull().sum().sum()
        print(f"✅ Missing values filled successfully!")
        print(f"   Original missing count: {original_missing}")
        print(f"   Final missing count: {final_missing}")
    
    def analyze_directors(self):
        """Task 2.1: Find and print the top 5 directors with the most titles on Netflix."""
        print("\n" + "="*60)
        print("🎬 TOP DIRECTORS ANALYSIS")
        print("="*60)
        
        # Split directors (some entries have multiple directors separated by commas)
        director_counts = {}
        for directors_str in self.df['director']:
            if pd.notna(directors_str) and directors_str != 'Unknown Director':
                directors = [d.strip() for d in directors_str.split(',')]
                for director in directors:
                    director_counts[director] = director_counts.get(director, 0) + 1
        
        # Get top 5 directors
        top_directors = sorted(director_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print("Top 5 Directors with most titles:")
        for i, (director, count) in enumerate(top_directors, 1):
            print(f"{i}. {director}: {count} titles")
        
        return top_directors
    
    def analyze_countries(self):
        """Task 2.2: Analyze countries and find top 5 with most titles."""
        print("\n" + "="*60)
        print("🌍 COUNTRIES ANALYSIS")
        print("="*60)
        
        # Split countries (some entries have multiple countries)
        country_counts = {}
        for countries_str in self.df['country']:
            if pd.notna(countries_str) and countries_str != 'Unknown Country':
                countries = [c.strip() for c in countries_str.split(',')]
                for country in countries:
                    country_counts[country] = country_counts.get(country, 0) + 1
        
        total_unique_countries = len(country_counts)
        print(f"Total unique countries: {total_unique_countries}")
        
        # Get top 5 countries
        top_countries = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print("\nTop 5 Countries with most titles:")
        for i, (country, count) in enumerate(top_countries, 1):
            print(f"{i}. {country}: {count} titles")
        
        return top_countries, total_unique_countries
    
    def analyze_content_types(self):
        """Task 2.3: Determine the number of movies versus TV shows."""
        print("\n" + "="*60)
        print("📺 CONTENT TYPE ANALYSIS")
        print("="*60)
        
        content_counts = self.df['type'].value_counts()
        
        print("Content Type Distribution:")
        for content_type, count in content_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"{content_type}: {count} titles ({percentage:.1f}%)")
        
        return content_counts
    
    def analyze_release_years(self):
        """Task 2.4: Find years with most and least titles released."""
        print("\n" + "="*60)
        print("📅 RELEASE YEAR ANALYSIS")
        print("="*60)
        
        year_counts = self.df['release_year'].value_counts().sort_index()
        
        most_titles_year = year_counts.idxmax()
        least_titles_year = year_counts.idxmin()
        
        print(f"Year with most titles: {most_titles_year} ({year_counts[most_titles_year]} titles)")
        print(f"Year with least titles: {least_titles_year} ({year_counts[least_titles_year]} titles)")
        
        return year_counts, most_titles_year, least_titles_year
    
    def setup_sentiment_analysis(self):
        """Task 3.1: Instantiate sentiment analysis pipeline."""
        print("\n" + "="*60)
        print("🤖 SETTING UP BERT SENTIMENT ANALYSIS")
        print("="*60)
        
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=-1  # Use CPU (-1) or GPU (0) if available
            )
            print("✅ BERT sentiment analysis pipeline created successfully!")
            return True
        except Exception as e:
            print(f"❌ Error setting up sentiment analysis: {e}")
            print("This might be due to missing dependencies. Please install transformers and torch.")
            return False
    
    def perform_sentiment_analysis(self):
        """Task 3.2-4: Perform sentiment analysis on descriptions and create sentiment_score column."""
        print("\n" + "="*60)
        print("💭 PERFORMING SENTIMENT ANALYSIS")
        print("="*60)
        
        if not self.sentiment_pipeline:
            print("❌ Sentiment analysis pipeline not available.")
            return False
        
        print("Analyzing sentiment for all titles...")
        sentiment_scores = []
        
        # Process descriptions in batches to avoid memory issues
        batch_size = 100
        total_titles = len(self.df)
        
        for i in range(0, total_titles, batch_size):
            batch_end = min(i + batch_size, total_titles)
            batch_descriptions = self.df['description'].iloc[i:batch_end].tolist()
            
            print(f"Processing batch {i//batch_size + 1}/{(total_titles + batch_size - 1)//batch_size}...")
            
            try:
                batch_results = self.sentiment_pipeline(batch_descriptions)
                
                for result in batch_results:
                    # Extract score from label (e.g., '5 stars' -> 5)
                    label = result['label']
                    score = int(label.split()[0])
                    sentiment_scores.append(score)
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Fill with neutral score if error occurs
                sentiment_scores.extend([3] * len(batch_descriptions))
        
        # Add sentiment score column
        self.df['sentiment_score'] = sentiment_scores
        
        print(f"✅ Sentiment analysis completed for {len(sentiment_scores)} titles!")
        print(f"Sentiment score range: {min(sentiment_scores)} - {max(sentiment_scores)}")
        
        return True
    
    def analyze_sentiment_results(self):
        """Task 3.5: Find top 10 highest and lowest sentiment movies."""
        print("\n" + "="*60)
        print("🏆 SENTIMENT ANALYSIS RESULTS")
        print("="*60)
        
        # Filter for movies only
        movies_df = self.df[self.df['type'] == 'Movie'].copy()
        
        if len(movies_df) == 0:
            print("No movies found in the dataset.")
            return
        
        # Top 10 highest sentiment movies
        top_movies = movies_df.nlargest(10, 'sentiment_score')[['title', 'sentiment_score', 'description']]
        
        print("🎬 Top 10 Movies with Highest Sentiment Score:")
        for i, (_, row) in enumerate(top_movies.iterrows(), 1):
            print(f"{i}. {row['title']} - Score: {row['sentiment_score']}")
            print(f"   Description: {row['description'][:100]}...")
            print()
        
        # Bottom 10 lowest sentiment movies
        bottom_movies = movies_df.nsmallest(10, 'sentiment_score')[['title', 'sentiment_score', 'description']]
        
        print("😔 Bottom 10 Movies with Lowest Sentiment Score:")
        for i, (_, row) in enumerate(bottom_movies.iterrows(), 1):
            print(f"{i}. {row['title']} - Score: {row['sentiment_score']}")
            print(f"   Description: {row['description'][:100]}...")
            print()
    
    def create_visualizations(self):
        """Task 4: Create stunning, interactive visualizations with Plotly."""
        print("\n" + "="*60)
        print("📊 CREATING INTERACTIVE VISUALIZATIONS")
        print("="*60)
        
        # 1. Content Type Distribution
        fig1 = px.pie(
            values=self.df['type'].value_counts().values,
            names=self.df['type'].value_counts().index,
            title="Netflix Content Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        fig1.show()
        
        # 2. Release Year Distribution
        year_counts = self.df['release_year'].value_counts().sort_index()
        fig2 = px.line(
            x=year_counts.index,
            y=year_counts.values,
            title="Netflix Titles by Release Year",
            labels={'x': 'Release Year', 'y': 'Number of Titles'},
            line_shape='linear'
        )
        fig2.update_layout(
            xaxis_title="Release Year",
            yaxis_title="Number of Titles",
            hovermode='x unified'
        )
        fig2.show()
        
        # 3. Top Countries
        country_counts = {}
        for countries_str in self.df['country']:
            if pd.notna(countries_str) and countries_str != 'Unknown Country':
                countries = [c.strip() for c in countries_str.split(',')]
                for country in countries:
                    country_counts[country] = country_counts.get(country, 0) + 1
        
        top_countries = dict(sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        fig3 = px.bar(
            x=list(top_countries.values()),
            y=list(top_countries.keys()),
            orientation='h',
            title="Top 10 Countries by Number of Netflix Titles",
            labels={'x': 'Number of Titles', 'y': 'Country'},
            color=list(top_countries.values()),
            color_continuous_scale='viridis'
        )
        fig3.show()
        
        # 4. Sentiment Score Distribution
        if 'sentiment_score' in self.df.columns:
            fig4 = px.histogram(
                self.df,
                x='sentiment_score',
                title="Distribution of Sentiment Scores",
                labels={'sentiment_score': 'Sentiment Score', 'count': 'Number of Titles'},
                nbins=10,
                color_discrete_sequence=['#FF6B6B']
            )
            fig4.update_layout(
                xaxis_title="Sentiment Score",
                yaxis_title="Number of Titles"
            )
            fig4.show()
            
            # 5. Sentiment vs Release Year
            fig5 = px.scatter(
                self.df,
                x='release_year',
                y='sentiment_score',
                color='type',
                title="Sentiment Score vs Release Year",
                labels={'release_year': 'Release Year', 'sentiment_score': 'Sentiment Score'},
                opacity=0.6
            )
            fig5.update_layout(
                xaxis_title="Release Year",
                yaxis_title="Sentiment Score"
            )
            fig5.show()
        
        print("✅ All visualizations created and displayed!")
    
    def run_complete_analysis(self):
        """Run the complete Netflix data analysis pipeline."""
        print("🚀 NETFLIX DATA EXPLORER CHALLENGE")
        print("="*60)
        
        # Task 1: Data Preprocessing and Cleaning
        print("\n📋 TASK 1: DATA PREPROCESSING AND CLEANING")
        print("-" * 40)
        
        if not self.load_data():
            return
        
        self.display_basic_info()
        missing_df = self.analyze_missing_values()
        self.fill_missing_values()
        
        # Task 2: Data Aggregation and Analysis
        print("\n📋 TASK 2: DATA AGGREGATION AND ANALYSIS")
        print("-" * 40)
        
        top_directors = self.analyze_directors()
        top_countries, total_countries = self.analyze_countries()
        content_counts = self.analyze_content_types()
        year_counts, most_year, least_year = self.analyze_release_years()
        
        # Task 3: Sentiment Analysis with BERT
        print("\n📋 TASK 3: SENTIMENT ANALYSIS WITH BERT")
        print("-" * 40)
        
        if self.setup_sentiment_analysis():
            if self.perform_sentiment_analysis():
                self.analyze_sentiment_results()
        
        # Task 4: Interactive Visualizations
        print("\n📋 TASK 4: INTERACTIVE VISUALIZATIONS")
        print("-" * 40)
        
        self.create_visualizations()
        
        print("\n🎉 ANALYSIS COMPLETE!")
        print("="*60)
        
        # Save processed data
        output_file = 'netflix_processed_data.csv'
        self.df.to_csv(output_file, index=False)
        print(f"💾 Processed data saved to: {output_file}")

def main():
    """Main function to run the Netflix Data Explorer."""
    # Create instance and run analysis
    explorer = NetflixDataExplorer()
    explorer.run_complete_analysis()

if __name__ == "__main__":
    main()