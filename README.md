# 📈 Yahoo Finance Stock Market Data Analysis & Prediction Project

![Project Thumbnail](https://img.shields.io/badge/Data-Science-blue) ![Python](https://img.shields.io/badge/Python-3.x-green) ![Pandas](https://img.shields.io/badge/Pandas-Data%20Cleaning-orange) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellow)

## 🎯 Project Overview

This project demonstrates a comprehensive data science workflow for analyzing and predicting stock market trends using Yahoo Finance data. The project focuses on cleaning extremely messy financial data and building machine learning models for price prediction and trading strategy development.

### Project Thumbnail

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   📊 YAHOO FINANCE STOCK ANALYSIS                           │
│                                                             │
│   Raw Data → Data Cleaning → Feature Engineering →         │
│   ML Models → Trading Strategy → Performance Analysis      │
│                                                             │
│   • 5,000+ records processed                               │
│   • Multiple data quality issues resolved                   │
│   • Regression & Classification models                      │
│   • Trading strategy backtesting                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Data Description](#-data-description)
- [Data Cleaning Process](#-data-cleaning-process)
- [Model Details](#-model-details)
- [Usage](#-usage)
- [Results](#-results)
- [Technologies Used](#-technologies-used)

## ✨ Features

- **Data Cleaning**: Handles extremely messy financial data with various formats and inconsistencies
- **Feature Engineering**: Creates technical indicators (EMA, MACD, RSI, ATR, Volatility)
- **Machine Learning Models**: 
  - Linear Regression for price prediction
  - Random Forest Regressor for price prediction
  - Random Forest Classifier for price direction prediction
- **Trading Strategy**: Implements moving average crossover strategy with backtesting
- **Visualization**: Generates charts for price trends, volume, volatility, and correlations
- **Performance Metrics**: Comprehensive evaluation including R², MAE, RMSE, and trading metrics

## 📁 Project Structure

```
data_science_project_yahoo_finance/
│
├── data_set/
│   └── extreme_messy_yahoo_finance.csv    # Raw messy data (5,000+ records)
│
├── clean_data_set/
│   └── clean_data_set.csv                 # Cleaned and processed data
│
├── model_code/
│   └── model.py.ipynb                     # Main analysis and modeling notebook
│
└── README.md                              # Project documentation
```

## 🔧 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Or install from requirements (if available):

```bash
pip install -r requirements.txt
```

## 📊 Data Description

### Raw Data Issues

The original dataset (`extreme_messy_yahoo_finance.csv`) contains:
- **Inconsistent date formats**: Multiple formats (DD-MM-YYYY, YYYY-MM-DD, DD/MM/YY, etc.)
- **Non-numeric values**: Text mixed with numbers (e.g., "209.23abc", "ERROR", "none")
- **Missing values**: NaN, empty strings, "?", "Null"
- **Special characters**: Parentheses, brackets, units (e.g., "7416863units")
- **Inconsistent text**: Mixed case, special symbols (↑, ↓, ???)

### Dataset Columns

- `date`: Trading date
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume
- `ma_10`: 10-day moving average
- `rsi`: Relative Strength Index
- `next_day_movement`: Price movement direction (Up/Down)
- `future_price`: Target variable for prediction

## 🧹 Data Cleaning Process

The cleaning pipeline includes:

1. **Date Standardization**
   - Parse multiple date formats
   - Convert to datetime objects
   - Forward fill missing dates

2. **Numeric Conversion**
   - Extract numeric values from mixed text
   - Handle special characters (parentheses, units, etc.)
   - Convert to appropriate data types

3. **Missing Value Treatment**
   - Identify missing values
   - Impute with median values for numeric columns
   - Forward/backward fill for categorical data

4. **Text Cleaning**
   - Standardize movement direction (Up/Down)
   - Remove special characters
   - Handle inconsistent formatting

5. **Data Validation**
   - Remove duplicates
   - Validate data ranges
   - Ensure data consistency

## 🤖 Model Details

### Feature Engineering

The project creates several technical indicators:

- **EMA (Exponential Moving Average)**: 12-day and 26-day EMAs
- **MACD (Moving Average Convergence Divergence)**: MACD line and signal line
- **RSI (Relative Strength Index)**: 14-period RSI calculation
- **ATR (Average True Range)**: 14-period ATR for volatility
- **Volatility**: Rolling standard deviation
- **Candle Body**: Absolute difference between open and close
- **Date Features**: Year, month, day extracted from dates

### Machine Learning Models

1. **Linear Regression**
   - Baseline model for price prediction
   - Uses standardized features

2. **Random Forest Regressor**
   - 200 estimators, max depth 10
   - Predicts future stock prices
   - Cross-validation for robustness

3. **Random Forest Classifier**
   - 200 estimators
   - Predicts price direction (up/down)
   - Classification metrics evaluation

### Trading Strategy

- **Moving Average Crossover**: 
  - Short MA (5-day) vs Long MA (20-day)
  - Buy signal when short MA > long MA
  - Sell signal when short MA < long MA
- **Backtesting**: Evaluates strategy performance
- **Metrics**: Total return, win ratio, strategy accuracy, drawdown

## 🚀 Usage

### Running the Analysis

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook model_code/model.py.ipynb
   ```

2. **Update File Paths** (if needed):
   - Ensure the data file path is correct in the first cell
   - The cleaned data will be saved automatically

3. **Execute Cells**:
   - Run cells sequentially to perform data cleaning
   - Execute analysis and modeling cells
   - View visualizations and results

### Data Cleaning Workflow

```python
# 1. Load messy data
df = pd.read_csv('data_set/extreme_messy_yahoo_finance.csv')

# 2. Clean dates
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# 3. Clean numeric columns (open, high, low, close, volume, etc.)
df['open'] = pd.to_numeric(df['open'], errors='coerce')
# ... similar for other columns

# 4. Handle missing values
df['open'] = df['open'].fillna(df['open'].median())

# 5. Save cleaned data
df.to_csv('clean_data_set/clean_data_set.csv')
```

## 📈 Results

### Data Cleaning Results
- **Original Records**: 5,000+ messy records
- **Cleaned Records**: 5,000 clean, validated records
- **Missing Values**: Handled through imputation
- **Data Quality**: Significantly improved

### Model Performance

The notebook includes comprehensive evaluation metrics:
- **R² Score**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **Accuracy**: Classification accuracy
- **Trading Metrics**: Returns, win ratio, drawdown

*Note: Specific results are generated when running the notebook with your data.*

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib**: Basic plotting
- **Seaborn**: Statistical visualizations
- **Scikit-learn**: Machine learning models
  - Linear Regression
  - Random Forest (Regressor & Classifier)
  - StandardScaler
  - Train/Test Split
  - Cross-validation
- **Jupyter Notebook**: Interactive development environment

## 📝 Notes

- The project demonstrates real-world data cleaning challenges
- All data cleaning steps are documented in the notebook
- Models can be further tuned for better performance
- Trading strategy can be extended with additional indicators

## 🔮 Future Enhancements

- [ ] Add more technical indicators
- [ ] Implement deep learning models (LSTM, GRU)
- [ ] Real-time data integration
- [ ] Web dashboard for visualization
- [ ] Automated trading signal generation
- [ ] Portfolio optimization strategies

## 👤 Author

**Narendra Kumar**

---

## 📄 License

This project is for educational and research purposes.

---

**⭐ If you find this project helpful, please consider giving it a star!**
