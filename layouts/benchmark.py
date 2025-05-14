from utils.functions import *
import plotly.express as px 
import plotly.graph_objects as go
# model training
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc

def model_benchmark():


    st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stButton>button { background-color: #dc143c; color: white; border-radius: 5px; }
        .stMetric { background-color: #040507; border-radius: 10px; padding: 10px; }
        .sidebar .sidebar-content { background-color: #f0f2f6; }
        h1, h2, h3 { color: #2c3e50; }
        .stPlotlyChart { 
            border: 2px solid #ddd;  /* Make border slightly thicker for visibility */
            border-radius: 5px; 
            padding: 15px;  /* Increase padding to create space inside the border */
            margin-left: 10px;  /* Add right margin to avoid container edge */
            overflow: hidden;  /* Prevent overflow */
            box-sizing: border-box;  /* Ensure padding is included in the element's size */
        }
    </style>
    """, unsafe_allow_html=True)



    st.title("📈 Customer Churn Prediction Dashboard")
    st.markdown("""
    Welcome to the **Churn Prediction Dashboard**! This tool showcases a **Logistic Regression model** for predicting customer churn in a telecom dataset (5,174 customers, 36.1% churn rate). Explore key metrics, feature importance, prediction comparisons, and business impact.

    **Key Highlights**:
    - **Model Performance**: 83.2% accuracy, 84.1% recall, 0.917 AUC.
    - **Top Features**: Tenure, total charges, monthly charges, month-to-month contracts.
    - **Business Impact**: Retains 314 customers, preserves $376,800, saves $31,400.
    """)

    st.markdown("Use the sidebar to navigate sections or adjust settings.")

    #datas
    df=pd.read_csv('data/churn_data_encoded.csv')
    raw_df = pd.read_csv('data/churn_data.csv')
    feat_imp = pd.read_csv('data/feat_imp.csv') 
    feat_imp = feat_imp.sort_values(by = 'Importance Mean', ascending=False).head(25)
    model = load_model()
    X = df.drop('churn_flag', axis=1)
    y=df['churn_flag'].values
   # Adjust if you have a separate test set
    x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
    y_true = df['churn_flag']
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    st.header("📊 Model Performance")
    st.markdown("The Logistic Regression model achieves **83.2% accuracy**, **84.1% recall**, and **0.917 AUC** on the test set.")

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "83.2%", help="Percentage of correct predictions.")
    with col2:
        st.metric("Recall", "84.1%", help="Percentage of actual churners correctly identified.")
    with col3:
        st.metric("AUC", "0.917", help="Area under the ROC curve, measuring model discrimination.")


    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["No Churn", "Churn"], columns=["Predicted No Churn", "Predicted Churn"])
    fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale='Reds', title="Confusion Matrix")
    fig_cm.update_layout(template="plotly_dark", font=dict(size=14), margin=dict(l=50, r=80, t=50, b=50))
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.3f})', line=dict(color='crimson')))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random'))
    fig_roc.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        margin=dict(l=50, r=50, t=50, b=50),
        yaxis_title="True Positive Rate",
        template="plotly_dark",
        font=dict(size=14)
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    st.header("🔍 Feature Importance")
    st.markdown("**The top features for classifying the churn propensity**.")

    # Feature Importance Plot
    feat_imp_top = feat_imp.sort_values(by='Importance Mean', ascending=False).head(25)

    fig = px.bar(
        x=feat_imp_top['Importance Mean'],
        y=feat_imp_top['Feature'],
        title='Top 25 Features Selected by Logistic Regression',
        color=feat_imp_top['Importance Mean'],
        color_continuous_scale=['#ffe5eb', '#f08080', '#dc143c']
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Importance Mean",
        yaxis_title="Feature",
        coloraxis_showscale=False,
        margin=dict(l=50, r=80, t=50, b=50),
        height=600,
        font=dict(size=14)
    )
    fig.update_yaxes(autorange='reversed')
    st.plotly_chart(fig, use_container_width=True)


    st.header("💼 Business Impact")

    total_customers = raw_df.shape[0]
    churners = raw_df.loc[raw_df['Churn'] == 1].shape[0]
    churn_rate = round((churners/total_customers)  *100, 2)
    recall = 0.84
    retention = int(churners * recall)
    campaing_retention = int(0.2 * retention) 
    cac_value = 100 * campaing_retention

    st.markdown(f"The model reduces churn, saving costs and preserving revenue for {total_customers} customers ({churn_rate}% churn rate).")

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Customers Retained", "314", "20% campaign success")
    with col2:
        st.metric("Revenue Preserved", "$376,800", "$1,200 CLV")
    with col3:
        st.metric("Cost Savings", "$31,400", "$100 acquisition cost")

    # Key Impacts
    st.subheader("Key Impacts")
    st.markdown(f"""
    -  **Improved Customer Retention**: With {churners} churned customers ({churn_rate}% of customers), the model correctly identifies approximately 84% of at-risk customers ({retention} customers). 
         A 20% successful retention campaign could retain {campaing_retention} customers.
    - **Cost Savings**: If the CAC(Customer Aquisition Cost) were \$100, the company would saved \${cac_value}.
    - **Operational Efficiency**: Automating predictions for {total_customers} customers reduces manual analysis time by an estimated 10 hours per week (assuming 1 hour per 700 customers manually reviewed), improving team productivity.
    """)

    # Retention Strategies
    with st.expander("Retention Strategies", expanded=True):
        st.markdown("""
        - **Month-to-Month Contracts (88% churn)**: Offer discounts to switch to one-year/two-year contracts.
        - **High Charges (\$74.50 vs. \$61.20)**: Provide affordable plans or bundles.
        - **Fiber Optic (33% churn)**: Improve service reliability and offer streaming bundles.
        - **Low Tenure (17.5 months)**: Enhance onboarding with free add-ons or loyalty incentives.
        """)
