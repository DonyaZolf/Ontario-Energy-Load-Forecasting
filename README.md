# Ontario Energy Load Forecasting Using Machine Learning (XGBoost)

### Project Overview
This project focuses on forecasting hourly electricity demand in Ontario using historical load data, weather information and engineered time-based features.  Energy load forecasting plays an important role in maintaining grid reliability, resource and energy allocation and reducing operational costs. I chose this project to deepen my understanding of time-series forecasting using modern ML techniques, and to explore the effectiveness of tree-based models, particularly XGBoost, in capturing seasonality and temporal patterns in time series data. 

### Data Description 
**Energy Demand Dataset:** Ontario hourly energy demand and pricing data from 2002 - 2023.
Link: https://www.kaggle.com/datasets/jacobsharples/ontario-electricity-demand

**Description:** The data is from a Kaggle dataset that compiled Ontatio's hourly demand and price, originally sourced from the Independent Electricity System Operator (IESO) public reports.

**Columns:** 
- `date`: calendar date (YYYY-MM-DD)
- `hour`: hour of the day (1-24)
- `hourly_demand`: total hourly Ontario energy demand in kWh
- `hourly_average_price`: average weighted hourly price in Canadian cents/kWh. 

**Weather data**:
- **Source**: Open-Mateo Historical Weather API. 
- **Details**: Hourly temperature (in celcius) in Toronto between 2002-2023, collected using download_weather_data.ipynb. 

*Both datasets are in the **data/** directory.*


### Methodology: 
**Data Cleaning and Visualization:**
- Constructed a continuous datetime index.
- Verified no missing timestamps or energy demand values. 
- Interpolated outlier demand points when necessary. 
- Converted hourly demand from kWh to MW. 
- Created initial exploratory visualizations to understand seasonal and daily patterns. 

**Feature Engineering:**
- Extracted **time-based features**: hour, dayofweek, month, dayofyear, and quarter. 
- Constructed **binary indicator** features:isweekend, isholiday (to capture energy demand changes on weekends and holidays). 
- Used **sine-cosine encoding** to represent cyclical time-based features (allows model to understand these features "wrap around"): hour, dayofweek, month. 
- Extracted **lag** and **rolling window** features to allow model to learn from historical context and capture patterns: lag_1h, lag_12h, lag_24h, lag_168h, rolling_6h, rolling_12h, rolling_24h and rolling_168h. 
- Added hourly temperature feature. 

**Training Models & Hyperparameter Tuning:**
- Chronological split of dataset into training and test sets: 
    - **Training set:** 2002-2018
    - **Test set:** 2019-2023. 
- Used TimeSeriesSplit (5 folds) with GridSearchCV for hyperparameter tuning. 
- Trained two models using training set: 
    - **Random Forest Regressor**(Baseline)
    - XGBoost Regressor

**Evaluating Models' Performance:**
- Evaluated performance of the models on the test set using various metrics: **MSE**, **RMSE**, **MAPE (%)**. 
- Plotted predicted energy demand values vs. actual values for both models. 
- Plotted models' residuals (errors) to analyze error behaviour. 

### Results 

| Model       |  MSE  |  RMSE  |  MAE  |  MAPE  |
| :---------- | :---: | :----: | :---: | :----: | 
| Random Forest Regressor (baseline model)| 123181.35 | 350.97 | 271.08 | 1.75% |
| XGBoost | 46527.49 | 215.70 | 157.21 | 1.02%|

- XGBoost performanced signfiicantly better than the baseline Random Forest model. 
    - XGBoost acheived nearly 40% lower RMSE than Random Forest. 
    - XGBoost had a MAPE of 1.02% indicating predictions are within ~1% of actual demand on average. 
- Visualizations show XGBoost captures peaks, troughs, and seasonal patterns more accurately.

### Tech Stack: 
**Languages**: Python 
**Libraries**: Pandas, NumPy, Scikit-learn, xgboost, Matplotlib.pyplot, Holidays, Requests 

### Next Steps: 
- Experiment with recurrent neural networks (LSTMs) for sequential modeling. 
- Build an interactive Streamlit dashboard to visualize forecasts

### How to run project:
1. Clone the repository:  
    git clone git@github.com:DonyaZolf/Ontario-Energy-Load-Forecasting.git
    cd Ontario-Energy-Load-Forecasting
2. Install dependencies: 
    pip install -r requirements .txt
3. Run the jupyter notebook. 
    jupyter notebook
