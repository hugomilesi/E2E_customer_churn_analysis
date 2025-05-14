import streamlit as st 
import numpy as np
import random
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



saved_model = pickle.load(open('utils/model.sav', 'rb')) 
df = pd.read_csv('data/churn_data.csv')

yes_no = ['No', 'Yes']
yes_no_service = ['No', 'Yes', 'No internet service']
yes_no_phone  = ['No phone service', 'Yes', 'No']
male_female = ['Male', 'Female']
payment = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
    'Credit card (automatic)']
dsl_fiber_no = ['DSL', 'Fiber optic', 'No']
contract_type = ['Month-to-month', 'One year', 'Two year']

tenure_val = 1.0
monthlycharges_val = 18.25
totalcharges_val = 0.0

def load_model():
    return pickle.load(open('utils/model.sav', 'rb')) 



def preprocess_data(input_data):
    
    columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges']
    
    df = pd.read_csv('data/churn_data.csv', usecols = columns) # loading dataset without the target('Churn') colmn
    data = {columns[i]:input_data[i] for i in range(len(columns))} # storing the new values into a dictionary
    data = pd.DataFrame(data, index=[0])
    pd.set_option('display.max_columns', 19)
    #df = df.append(data, ignore_index = True) # adding the dict values into the dataframe
    df = pd.concat([df, data], ignore_index = True)
    mms = MinMaxScaler() # normalization
    
    # splitting categorical columns from numeric for encoding
    categorical = df.select_dtypes(include = ['object']).columns
    quant = df.select_dtypes(include = ['float64', 'int64']).columns # only numeric values
    
    numeric = df[quant]
    encoded = pd.get_dummies(df[categorical], dtype=int)

    # normalizing numeric data
    for num in quant:
        numeric[num] = mms.fit_transform(numeric[num].values.reshape(-1, 1))

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
   
    return prediction[0], prediction_proba

def randomizer():
        
    global yes_no_service
    yes_no_service = random.sample(yes_no_service, 3)
    
    global yes_no_phone
    yes_no_phone = random.sample(yes_no_phone, 3)
    
    global yes_no
    yes_no = random.sample(yes_no, 2)
    
    global male_female
    male_female = random.sample(male_female, 2)
    
    global payment
    payment = random.sample(payment, len(payment))
    
    global dsl_fiber_no
    dsl_fiber_no = random.sample(dsl_fiber_no, len(dsl_fiber_no))
    
    global contract_type
    contract_type = random.sample(contract_type, len(contract_type))
    
    global tenure_val
    tenure_val = random.randint(1, 72)
    
    global monthlycharges_val
    monthlycharges_val = round(random.uniform(18.25, 118.75), 2)
    
    global totalcharges_val
    totalcharges_val = round(random.uniform(18.8, 8684.8), 2)
    