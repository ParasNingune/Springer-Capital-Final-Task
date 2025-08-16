# Employee Email Sentiment Analysis - Enhanced Version

## Summary
This project provides a comprehensive sentiment analysis tool for employee emails, featuring both simple and advanced analytics. The system uses AI models to classify sentiment, create detailed visualizations, detect employees who might need support, and generate predictive insights.

## 🆕 New Features & Improvements
- **Simplified Script Architecture**: Clean, step-by-step code structure for easy understanding
- **Enhanced Visualizations**: 10+ professional charts including employee rankings and risk analysis
- **Advanced Risk Detection**: Sliding window analysis for identifying concerning patterns
- **Predictive Modeling**: Machine learning models with feature importance analysis
- **Comprehensive Reporting**: Automated report generation with actionable insights
- **Flexible Configuration**: Adjustable confidence thresholds and batch processing

---

## Top Employees

### **Top 5 Positive Employees**
- `patti.thompson@enron.com` —> **101**
- `sally.beck@enron.com` —> **84**
- `john.arnold@enron.com` —> **75**

### **Top 3 Negative Employees**
- `lydia.delgado@enron.com` —> **99**
- `don.baughman@enron.com` —> **85**
- `john.arnold@enron.com` —> **84**

---

## Flight Risk Employees
Employees flagged as **flight risks** (≥4 negative messages in any rolling 30-day window):
- `lydia.delgado@enron.com`
- `don.baughman@enron.com`
- `john.arnold@enron.com.com`
- `patti.thompson@enron.com.com`
- `johnny.palmer@enron.com`

---

## Key Insights from Enhanced Analysis
1. **AI-Powered Classification**: Uses state-of-the-art transformer models with confidence scoring
2. **Comprehensive Visualization Suite**: Creates 10+ charts covering basic trends to advanced analytics
3. **Risk Detection Algorithm**: Identifies employees with concerning negative sentiment patterns
4. **Predictive Capabilities**: Machine learning models predict sentiment based on communication patterns
5. **Professional Reporting**: Generates executive-ready summaries and recommendations

## Advanced Analytics Features
- 🌟 **Monthly Employee Rankings**: Top positive contributors and employees needing support
- 📊 **Predictive Modeling**: Linear regression with actual vs predicted visualizations
- 🔍 **Email Pattern Analysis**: Correlation between email length, frequency, and sentiment
- 🚨 **Enhanced Risk Detection**: Timeline visualization of concerning communication patterns
- 📈 **Feature Importance**: Understanding what factors drive sentiment predictions

---

## Recommendations for Implementation
- **Deploy Monthly Monitoring**: Set up automated sentiment tracking for early intervention
- **Focus on High-Risk Employees**: Proactively support employees showing negative patterns
- **Leverage Positive Contributors**: Engage top-scoring employees as culture champions
- **Data-Driven HR Decisions**: Use confidence scores and model predictions for targeted initiatives
- **Expand Data Sources**: Integrate additional communication channels for comprehensive analysis
- **Regular Model Updates**: Retrain models with new data to maintain accuracy

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ParasNingune/Springer-Capital-Final-Task.git
cd Springer-Capital-Final-Task
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Data
- Place your email data CSV file in the `Data` folder
- Ensure the CSV has columns: `body`, `date`, `from`
- Default input file: `./Data/test(in).csv`

---

## How to Run the Analysis

### Option 1: Simple Python Script (Recommended)
```bash
python script.py
```
This runs the complete analysis pipeline with all enhanced features.

### Option 2: Interactive Jupyter Notebook
```bash
jupyter notebook
```
Then open and run `updated-notebook.ipynb` for step-by-step execution.


## Analysis Pipeline

The enhanced sentiment analysis follows these steps:

### 1. **Data Loading & Cleaning** 📧
   - Loads email data from CSV with robust error handling
   - Validates required columns and data integrity
   - Removes duplicates and invalid entries
   - Converts dates and handles missing values

### 2. **AI Model Setup** 🤖
   - Loads pre-trained sentiment classification models
   - Primary: `nlptown/bert-base-multilingual-uncased-sentiment` (1-5 star ratings)
   - Validator: Cross-validation for confidence assessment
   - Automatic model testing and validation

### 3. **Sentiment Classification** 🎯
   - Batch processing for efficiency (configurable batch size)
   - Confidence-based decision making (adjustable threshold)
   - Conservative approach for low-confidence predictions
   - Converts star ratings to sentiment labels

### 4. **Comprehensive Visualizations** 📊
   - **Basic Charts**: Distribution, trends, top senders
   - **Advanced Analytics**: Employee rankings, predictive models
   - **Risk Analysis**: Timeline visualization, pattern detection
   - **Performance Metrics**: Model accuracy and feature importance

### 5. **Risk Detection & Employee Analysis** ⚠️
   - Sliding window analysis for concerning patterns
   - Identifies employees needing support
   - Monthly performance rankings
   - Timeline visualization of negative sentiment clusters

### 6. **Predictive Modeling** 🔮
   - Linear regression for sentiment prediction
   - Feature importance analysis
   - Actual vs predicted visualizations with trend lines
   - Model performance metrics (R², MSE)

### 7. **Report Generation** 📄
   - Automated summary reports
   - Executive-ready insights and recommendations
   - Detailed CSV output with all metrics
   - Professional visualization export

## Configuration Options

Edit these settings in the script for customization:

```python
# Confidence threshold (0.0 to 1.0)
CONFIDENCE_THRESHOLD = 0.25

# Processing batch size
BATCH_SIZE = 16

# File paths
INPUT_FILE = "./Data/test(in).csv"
OUTPUT_FILE = "./Data/test_out_processed.csv"
```

## Project Structure
```bash
.
├── script.py                          # Enhanced main analysis script
├── simple_sentiment_analysis.py       # Streamlined version with all features
├── notebook.ipynb                     # Previous notebook version
├── updated-notebook.ipynb             # Enhanced notebook version
├── requirements.txt                   # Python dependencies
├── final-report.pdf                   # Comprehensive analysis report
├── README.md                          # This file
├── Visualization/                     # Generated charts and graphs
│   ├── Sentiment Distribution.png
│   ├── Monthly Sentiment Trends.png
│   ├── Top 3 Positive Employee Per Month.png
│   ├── Actual vs Predicted Score.png
│   ├── Feature Importance.png
│   ├── Email Length vs Confidence.png
│   └── Risk Timeline.png
└── Data/                             # Input and output data
    ├── test(in).csv                  # Input email data
    ├── test_out_processed.csv        # Processed results
    └── summary_report.txt            # Analysis summary
```

## Output Files

After running the analysis, you'll find:

### 📊 Visualizations (./Visualization/)
- **Basic Charts**: Sentiment distribution, monthly trends, top senders
- **Advanced Analytics**: Employee rankings, predictive models, risk analysis
- **All charts saved at 300 DPI** for professional presentations

### 📄 Reports (./Data/)
- **Detailed CSV**: Complete analysis with confidence scores and metrics
- **Summary Report**: Executive summary with key insights and recommendations
- **Model Performance**: Predictive accuracy metrics and feature importance

### 🎯 Key Insights Available
- Sentiment distribution across all communications
- Monthly trends and seasonal patterns
- Employee performance rankings and risk flags
- Predictive model accuracy and feature importance
- Actionable recommendations for HR initiatives

## Troubleshooting

### Common Issues:
1. **Missing CSV columns**: Ensure your data has `body`, `date`, `from` columns
2. **Model download**: First run requires internet connection for AI models
3. **Memory issues**: Reduce `BATCH_SIZE` if experiencing memory problems
4. **Date parsing**: Check date format compatibility

### Support:
- Check the console output for detailed error messages
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify input file path and format
- For additional help, review the error logs generated during execution

