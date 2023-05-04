import streamlit as st 
import numpy as np
import random
import pickle
import functions
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


saved_model = pickle.load(open('model_xgb.sav', 'rb')) 
df = pd.read_csv('churn_data.csv')
#st.set_page_config(page_title='Churn analysis', layout = 'wide', initial_sidebar_state = 'auto')



def main():
    st.set_page_config(page_title='Churn analysis', layout = 'wide', initial_sidebar_state = 'auto')
    
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
            functions.contract_type
        )
    with col2:
        internetservice = st.selectbox(
            'Internet service',
            functions.dsl_fiber_no
            )
    with col3:
        paymentmethod = st.selectbox(
            'Customer has payment method?',
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
        
        tenure = st.number_input('Tenure(months)', value = functions.tenure_val)
        monthlycharges = st.number_input('Customer monthly charges.', value = functions.monthlycharges_val)
        totalcharges = st.number_input('Customer total charge.', value = functions.totalcharges_val)
        
    diagnosis = ''
    
    #button for prediction
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    if col3.button('MAKE PREDICTION'):
        
        churn_pred = functions.model_prediction([gender, seniorcitizen, partner, dependents, tenure, phoneservice, multiplelines, 
                                internetservice, onlinesecurity, onlinebackup, deviceprotection, devicesupport, streamingtv,
                                streamingmovies, contract, paperlessbilling, paymentmethod, monthlycharges, totalcharges
                                ])
        
        if churn_pred[0] == 1:
            st.warning(':warning: This customer is close to churn. :warning:')
            st.warning(f" :boom: Propensity to churn: {churn_pred[1]}")
        else:
            st.success(":sparkles: This customer isn't close to churn. :sparkles:")
            st.success(f' :boom: Propensity to churn: {churn_pred[1]}%')  

    if col4.button('RANDOMIZE!'):
        functions.randomizer()

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    

    