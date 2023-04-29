import random
import streamlit as st


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
    