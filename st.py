from layouts.new_data import test_data
from layouts.benchmark import model_benchmark, load_model
import streamlit as st




def main():
    st.set_page_config(
        page_title='Churn analysis',
        page_icon = "🧊",
        layout = 'wide',
        initial_sidebar_state = 'auto',
        menu_items={
        'Report a bug': "mailto:hugogmilesi@gmail.com",
    })
    
    st.sidebar.title('📊 Navigation')
    st.sidebar.markdown("Navigate through the sections to explore the Logistic Regression model for customer churn prediction.")
    options = st.sidebar.radio('Pages', options = ['Test New Data', 'Model Details' ])
    
    st.header('Customer Churn Prediction')
    st.markdown("""
                ## About this project.
                - This is a visual representation of a data mining project using [telco customer churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) stored at kaggle. 
                - The model of choosing was **Logistic Regression**. If you want to checkout how i trained the model, please take a look at this [notebook](https://github.com/hugomilesi/E2E_customer_churn_analysis/blob/main/data_mining.ipynb) here.
                - The model will predict if a customer will churn based on the data you provide below.
                """)
                
    
    if options == 'Test New Data':
        test_data()
    elif options  == 'Model Details':
        model_benchmark()
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    

    