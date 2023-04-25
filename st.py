import streamlit as st 
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler

saved_model = pickle.load(open('model.sav', 'rb')) 
df = pd.read_csv('churn_data.csv')

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
    
    print(df)
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
    print(prediction)
    
    if(prediction[0] == 0):
        return "This customer doesn't seem close to churn."
    else:
        return "This customer is close to churn!"
    
    
def main():
    
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
    internetservice = st.selectbox(
        'Internet service',
        ['DSL', 'Fiber optic', 'No']
        )
    onlinesecurity = st.selectbox(
        'Customer has online security?',
        ['No', 'Yes', 'No internet service']
        )
    onlinebackup = st.selectbox(
        'Customer has onlineBackup?',
        ['No', 'Yes', 'No internet service']
        )
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
    contract = st.selectbox(
        'Customer contract type',
        ['Month-to-month', 'One year', 'Two year']
    )
    paperlessbilling = st.selectbox(
        'Customer has paperless billing benefit?',
        ['No', 'Yes']
    )
    paymentmethod = st.selectbox(
        'Customer has payment method?',
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
       'Credit card (automatic)']
       )
    tenure = st.number_input('How many months the customer stayed with the signature?')
    monthlycharges = st.number_input('Customer monthly charges.')
    totalcharges = st.number_input('Customer total charge.')
    
    diagnosis = ''
    
    #button for prediction
    if st.button('Customer Results'):
        
        churn_pred = model_prediction([gender, seniorcitizen, partner, dependents, tenure, phoneservice, multiplelines, 
                                internetservice, onlinesecurity, onlinebackup, deviceprotection, devicesupport, streamingtv,
                                streamingmovies, contract, paperlessbilling, paymentmethod, monthlycharges, totalcharges
                                ])
        
        st.success(churn_pred)
    

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    

    