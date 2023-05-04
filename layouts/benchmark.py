from functions import *
import plotly.express as px 

def model_benchmark():
    df = pd.read_csv('data/feat_imp.csv') 
    df.columns = ['feature', 'score']
    df = df.sort_values(by = 'score', ascending=True)
    
    st.header('Model Benchmark')
    fig=px.bar(x = df['score'], y = df['feature'], title = '(RF) Feature Importances')
    st.write(fig)
    
