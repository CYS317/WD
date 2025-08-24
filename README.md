# Netflix Data Explorer Challenge 🚀

A comprehensive analysis of Netflix titles using BERT sentiment analysis and interactive visualizations with Plotly.

## 🎯 Challenge Overview

This project explores the `netflix_titles.csv` dataset through:
- **Data Preprocessing & Cleaning**: Handle missing values and data quality issues
- **Data Aggregation & Analysis**: Discover insights about directors, countries, content types, and release years
- **BERT Sentiment Analysis**: Analyze movie descriptions using pre-trained BERT models
- **Interactive Visualizations**: Create stunning charts with Plotly

## 📋 Tasks Completed

### Task 1: Data Preprocessing and Cleaning
- ✅ Load `netflix_titles.csv` into Pandas DataFrame
- ✅ Display basic DataFrame information
- ✅ Identify and calculate missing values per column
- ✅ Fill missing values appropriately

### Task 2: Data Aggregation and Analysis
- ✅ Find top 5 directors with most titles
- ✅ Count unique countries and find top 5
- ✅ Determine movies vs TV shows distribution
- ✅ Find years with most/least titles released

### Task 3: Sentiment Analysis with BERT
- ✅ Instantiate BERT sentiment analysis pipeline
- ✅ Analyze descriptions for sentiment scores
- ✅ Create `sentiment_score` column
- ✅ Find top/bottom 10 movies by sentiment

### Task 4: Interactive Visualizations
- ✅ Content type distribution pie chart
- ✅ Release year timeline
- ✅ Top countries horizontal bar chart
- ✅ Sentiment score distribution histogram
- ✅ Sentiment vs release year scatter plot

## 🛠️ Installation

1. **Clone or download the project files**
2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Netflix dataset:**
   - Place `netflix_titles.csv` in the same directory as the script
   - Or modify the `csv_path` parameter in the code

## 🚀 Usage

### Quick Start
```bash
python netflix_data_explorer.py
```

### Custom Usage
```python
from netflix_data_explorer import NetflixDataExplorer

# Create explorer instance
explorer = NetflixDataExplorer('path/to/your/netflix_titles.csv')

# Run specific tasks
explorer.load_data()
explorer.analyze_directors()
explorer.create_visualizations()

# Or run complete analysis
explorer.run_complete_analysis()
```

## 📊 Expected Output

The script will generate:
- **Console output**: Detailed analysis results with emojis and formatting
- **Interactive plots**: Plotly visualizations that open in your browser
- **Processed data**: `netflix_processed_data.csv` with sentiment scores

## 🔧 Key Features

- **Batch Processing**: Handles large datasets efficiently
- **Error Handling**: Graceful fallbacks for missing dependencies
- **Memory Management**: Processes sentiment analysis in batches
- **Flexible Input**: Customizable CSV file path
- **Professional Output**: Clean, formatted console output

## 📈 Sample Visualizations

1. **Content Distribution**: Pie chart showing Movies vs TV Shows
2. **Release Timeline**: Line chart of titles by year
3. **Country Rankings**: Horizontal bar chart of top countries
4. **Sentiment Analysis**: Histogram of sentiment scores
5. **Temporal Sentiment**: Scatter plot of sentiment vs release year

## ⚠️ Important Notes

- **BERT Model**: Uses `nlptown/bert-base-multilingual-uncased-sentiment`
- **Memory Usage**: Sentiment analysis can be memory-intensive for large datasets
- **Dependencies**: Requires `transformers` and `torch` for BERT functionality
- **Browser**: Plotly visualizations open in your default web browser

## 🐛 Troubleshooting

### Common Issues:
1. **Missing CSV file**: Ensure `netflix_titles.csv` is in the correct location
2. **Dependencies**: Run `pip install -r requirements.txt`
3. **Memory errors**: The script processes data in batches to avoid memory issues
4. **BERT download**: First run may take time to download the pre-trained model

### Performance Tips:
- Use GPU if available (modify `device` parameter in `setup_sentiment_analysis`)
- Reduce batch size for memory-constrained systems
- Process only necessary columns for large datasets

## 📝 Data Schema

Expected CSV columns:
- `show_id`: Unique identifier
- `type`: Movie or TV Show
- `title`: Title of the content
- `director`: Director(s) name(s)
- `cast`: Cast member names
- `country`: Country of origin
- `date_added`: Date added to Netflix
- `release_year`: Year of release
- `rating`: Content rating
- `duration`: Length of content
- `listed_in`: Genre categories
- `description`: Content description

## 🤝 Contributing

Feel free to enhance the analysis with:
- Additional visualization types
- More sophisticated sentiment analysis
- Genre analysis and clustering
- Recommendation systems
- Export to different formats

## 📄 License

This project is created for educational purposes as part of the Netflix Data Explorer Challenge.

---

**Happy Data Exploring! 🎬📊✨**