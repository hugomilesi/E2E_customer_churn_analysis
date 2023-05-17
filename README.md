
# E2E_customer_churn_analysis
- This project aims to train and deploy a machine learning model for classifying customer churns.
- It goes since data collection to deploying model into production.
- Optimized Random Forest, XGBoost, Logistic Regression, Naive Bayes and KNN using GridsearchCV to reach the best model.
- Built a client facing API using streamlit.

# Resources Used
**Python Version:** 3.10<br>
**Packages:** Streamlit, Sklearn, matplotlib, seaborn, pickle<br>
**For Web Framework Requirements:** ```pip install -r requirements.txt```<br>
**Run** ```streamlit run st.py ```<br>

# Data Cleaning
- Removed NaN rows.
- Renamed row and column values for better understanding.
- Transformed some variables to the right format.

# EDA
### Some hightlights from the tables
- I built a chart using tenure(monthly) column to check the distribution types between churns.
- Calculate the churn ratio and made a pie chart.
- 

<div style="display: flex;">
  <img src="img/churn_pie.png" alt="Alt Text" width="300" height="auto" style="flex: 1;">
  <img src="img/churn_distribution.png" alt="Alt Text" width="400" height="auto" style="flex: 1;">
</div>


## Instalation
1. Clone this repo
2. ```pip install -r requirements.txt```
3. ```streamlit run st.py ```

## Or check the web version 
- just visit [This link](https://hugomilesi-e2e-customer-churn-analysis-st-iguvbo.streamlit.app) and insert new data.



