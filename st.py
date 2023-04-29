import streamlit as st 
import numpy as np
import random
import pickle
import functions
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


saved_model = pickle.load(open('model_xgb.sav', 'rb')) 
df = pd.read_csv('churn_data.csv')
st.set_page_config(page_title='Churn analysis', layout = 'wide', initial_sidebar_state = 'auto')






def show_data(input_data):
    
    columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges']
    
    df = pd.read_csv('churn_data.csv', usecols = columns) # loading dataset
    data = {columns[i]:input_data[i] for i in range(len(columns))}
    
    pd.set_option('display.max_columns', 19)
    df = df.append(data, ignore_index = True)
    
    return st.dataframe(df)


def preprocess_data(input_data):
    
    columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges']
    
    df = pd.read_csv('churn_data.csv', usecols = columns) # loading dataset without the target('Churn') colmn
    data = {columns[i]:input_data[i] for i in range(len(columns))} # storing the new values into a dictionary
    pd.set_option('display.max_columns', 19)
    df = df.append(data, ignore_index = True) # adding the dict values into the dataframe
    
    mms = MinMaxScaler() # normalization
    
    for i in range(len(columns)):
        data = {columns[i]:input_data[i]}
    
    # splitting categorical columns from numeric for encoding
    categorical = df.select_dtypes(include = ['object']).columns
    quant = df.select_dtypes(include = ['float64', 'int64']).columns # only numeric values
    
    numeric = df[quant]
    encoded = pd.get_dummies(df[categorical])

    # normalizing numeric data
    for num in quant:
        df[num] = mms.fit_transform(numeric[num].values.reshape(-1, 1))

    # merging the encoded values with numerics again
    all_data = pd.concat([encoded, numeric], axis = 1)

    return all_data.iloc[-1, :] # using the last value inserted into the dataframe
    

def model_prediction(input_data):
    input_data = preprocess_data(input_data).values
    input_data_asarray = np.asarray(input_data)
    input_data_reshaped = input_data_asarray.reshape(1, -1)

    prediction = saved_model.predict(input_data_reshaped)
    prediction_proba = saved_model.predict_proba(input_data_reshaped)
    prediction_proba = float(prediction_proba[:, 1] * 100)
    prediction_proba = round(prediction_proba, 2)
    
    if(prediction[0] == 0):
        return f"""This customer doesn't seem close to churn. Churn Probability: {prediction_proba}%"""
    else:
        return f"""This customer is close to churn! Churn probability: {prediction_proba}%"""
    
    
def main():
    
    st.header('XGBoost Customer Churn Prediction')
    st.markdown("""
                ## About this project.
                - This is a visual representation of a data mining project using [telco customer churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) stored at kaggle. 
                - The model of choosing was **XGBoost**. If you want to checkout how i trained the model, please take a look at this [notebook](https://github.com/hugomilesi/E2E_customer_churn_analysis/blob/main/data_mining.ipynb) here.
                - The model will predict if a customer will churn based on the data you provide below.
                """)
    
    
    # divider
    st.markdown("""<hr style = 'border-top:8px solid crimson; border-radius:5px'>""", unsafe_allow_html=True)
    
    st.markdown("""
                ## Most relevant fields
                - Most relevant fields accordingly to the trained model.
                - Information inserted here will have more impact in model's decision than the rest of the fields.
                """)
    col1, col2, col3 = st.columns(3)
    with col1:
        contract = st.selectbox(
            'Customer contract type',
            #['Month-to-month', 'One year', 'Two year']
            functions.contract_type
        )
    with col2:
        internetservice = st.selectbox(
            'Internet service',
            #['DSL', 'Fiber optic', 'No']
            functions.dsl_fiber_no
            )
    with col3:
        paymentmethod = st.selectbox(
            'Customer has payment method?',
            #['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
        #'Credit card (automatic)']
            functions.payment
    )
    # divider
    st.markdown("""<hr style = 'border-top:8px solid crimson; border-radius:5px'>""", unsafe_allow_html=True)
    st.markdown("""## Least Relevant Fields""")
    col1, col2, col3, col4 = st.columns(4)
    # column 1
    with col1:
        gender = st.selectbox(
        'Gender',
        functions.male_female
        )
        seniorcitizen = st.selectbox(
        'Senior citizen?',
        functions.yes_no
        )
        partner = st.selectbox(
        'Partner?',
        functions.yes_no
        )
        dependents = st.selectbox(
        'Dependents?',
        functions.yes_no
        )
    with col2:
        phoneservice = st.selectbox(
        'Phone service?',
        functions.yes_no
        )
        multiplelines = st.selectbox(
        'Multiple lines?',
        functions.yes_no_phone
        )
        onlinesecurity = st.selectbox(
        'online security',
        functions.yes_no_service
        )
        onlinebackup = st.selectbox(
        'Online Backup?',
        functions.yes_no_service
        )
    # column 2
    with col3:
        deviceprotection = st.selectbox(
        'Device protection?',
        functions.yes_no_service
        )
        devicesupport = st.selectbox(
            'Device support?',
            functions.yes_no_service
            )
        streamingtv = st.selectbox(
            'Streaming tv?',
            functions.yes_no_service
            )
        streamingmovies = st.selectbox(
            'Streaming movies benefit?',
            functions.yes_no_service
    
            )
    with col4:
        paperlessbilling = st.selectbox(
            'Paperless billing benefit?',
            functions.yes_no
        )
        
       
        tenure = st.number_input('Tenure(months)', value = functions.tenure_val, min_value = 1.0, max_value = 72.0)
        monthlycharges = st.number_input('Customer monthly charges.', value = functions.monthlycharges_val, max_value = 118.75)
        totalcharges = st.number_input('Customer total charge.', value = functions.totalcharges_val, max_value = 8684.8)
        
    diagnosis = ''
    
    #button for prediction
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    if col3.button('MAKE PREDICTION'):
        
        churn_pred = model_prediction([gender, seniorcitizen, partner, dependents, tenure, phoneservice, multiplelines, 
                                internetservice, onlinesecurity, onlinebackup, deviceprotection, devicesupport, streamingtv,
                                streamingmovies, contract, paperlessbilling, paymentmethod, monthlycharges, totalcharges
                                ])
        
        st.success(churn_pred)

    if col4.button('RANDOMIZE!'):
        functions.randomizer()
    

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    

    