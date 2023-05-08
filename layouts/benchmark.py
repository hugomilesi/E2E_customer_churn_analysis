from functions import *
import plotly.express as px 

def model_benchmark():
    df = pd.read_csv('data/feat_imp.csv') 
    df.columns = ['feature', 'score']
    df = df.sort_values(by = 'score', ascending=True)
    
    st.header('Model Benchmark')
    st.markdown("""
                - Here You can see the most important features selected by the Random Forests model.
                
                """)
    fig=px.bar(x = df['score'], y = df['feature'], title = '(RF) Feature Importances', width=800, height=800)
    st.write(fig)
    
