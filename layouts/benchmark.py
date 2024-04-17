from utils.functions import *
import plotly.express as px 

def model_benchmark():
    df = pd.read_csv('data/feat_imp.csv') 
    df = df.sort_values(by = 'score', ascending=True)

    
    st.header('Model Benchmark')
    st.markdown("""
                - Here You can see the most important features selected by the XGB model.
                
                """)
    fig=px.bar(x = df['score'], y = df['feature'], title = '(XGB) Feature Importances', width=1200, height=800)
    fig.update_traces(marker_color = 'crimson')
    st.write(fig)
    
    st.header('Validation Data Test Results')
    # Tableau DashBoard
    iframe = '<iframe src="https://public.tableau.com/views/Churn_analysis_16878380667720/Dashboard?:showVizHome=no&:embed=true:language=pt-BR" width="900" height="700" allowfullscreen></iframe>'
    st.markdown(iframe, unsafe_allow_html=True)
