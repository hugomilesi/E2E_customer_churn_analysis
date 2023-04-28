import streamlit as st 
import numpy as np
import pickle
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
    for num in numeric:
        df[num] = mms.fit_transform(numeric[num].values.reshape(-1, 1))

    # merging the encoded values with numerics again
    all_data = pd.concat([encoded, numeric], axis = 1)

    return all_data.iloc[-1, :] # using the last value inserted into the dataframe
    

def model_prediction(input_data):
    
    input_data = preprocess_data(input_data).values
    input_data_asarray = np.asarray(input_data)
    input_data_reshaped = input_data_asarray.reshape(1, -1)

    prediction = saved_model.predict(input_data_reshaped)
    
    if(prediction[0] == 0):
        return "This customer doesn't seem close to churn."
    else:
        return "This customer is close to churn!"
    
    
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
    
    contract = st.selectbox(
        'Customer contract type',
        ['Month-to-month', 'One year', 'Two year']
    )
    internetservice = st.selectbox(
        'Internet service',
        ['DSL', 'Fiber optic', 'No']
        )
    paymentmethod = st.selectbox(
        'Customer has payment method?',
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
    'Credit card (automatic)']
    )
    # divider
    st.markdown("""<hr style = 'border-top:8px solid crimson; border-radius:5px'>""", unsafe_allow_html=True)
    st.markdown("""## Least Relevant Fields""")
    col1, col2 = st.columns(2)
    # column 1
    with col1:
        gender = st.selectbox(
        'Gender',
        ['Male', 'Female']
        )
        seniorcitizen = st.selectbox(
        'Customer is Senior?',
        ['Yes', 'No']
        )
        partner = st.selectbox(
        'Customer has a partner?',
        ['Yes', 'No']
        )
        dependents = st.selectbox(
        'Customer has a dependents?',
        ['Yes', 'No']
        )
        phoneservice = st.selectbox(
        'Customer has phone service?',
        ['Yes', 'No']
        )
        multiplelines = st.selectbox(
        'Customer has multiple lines?',
        ['No phone service', 'Yes', 'No']
        )
        onlinesecurity = st.selectbox(
        'Customer has online security?',
        ['No', 'Yes', 'No internet service']
        )
        onlinebackup = st.selectbox(
        'Customer has onlineBackup?',
        ['No', 'Yes', 'No internet service']
        )
    # column 2
    with col2:
        deviceprotection = st.selectbox(
        'Customer has device protection?',
        ['No', 'Yes', 'No internet service']
        )
        devicesupport = st.selectbox(
            'Customer has device support?',
            ['No', 'Yes', 'No internet service']
            )
        streamingtv = st.selectbox(
            'Customer has streaming tv?',
            ['No', 'Yes', 'No internet service']
            )
        streamingmovies = st.selectbox(
            'Customer has streaming movies benefit?',
            ['No', 'Yes', 'No internet service']
            )
        paperlessbilling = st.selectbox(
            'Customer has paperless billing benefit?',
            ['No', 'Yes']
        )
        tenure = st.number_input('Tenure(months)')
        monthlycharges = st.number_input('Customer monthly charges.')
        totalcharges = st.number_input('Customer total charge.')
    
    
    
    diagnosis = ''
    
    #button for prediction
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    if col3.button('CUSTOMER DIAGNOSIS'):
        
        churn_pred = model_prediction([gender, seniorcitizen, partner, dependents, tenure, phoneservice, multiplelines, 
                                internetservice, onlinesecurity, onlinebackup, deviceprotection, devicesupport, streamingtv,
                                streamingmovies, contract, paperlessbilling, paymentmethod, monthlycharges, totalcharges
                                ])
        
        st.success(churn_pred)
    

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    

    