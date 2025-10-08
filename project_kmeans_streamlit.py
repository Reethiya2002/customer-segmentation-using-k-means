import streamlit as st
import pandas as pd 
import joblib

#load saved model and scaler
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

#define cluster names(optional)
cluster_labels = {
    0: 'bugdet customers',
    1: 'standard customers',
    2: 'target customers (high income, high spend)',
    3: 'potential customers (high income, low spend)',
    4: 'low income, high spend'    
}

#streamlit UI
st.title('Customer Segmentation using KMeans')
st.markdown('enter new customer details to predict the segment')

# user inputs
income = st.number_input('Annual Income (k$)', min_value=10, max_value=150, value=50)
spending = st.number_input('Spending Score (1-100)', min_value=1, max_value=100, value=50)

#predict cluster
if st.button('Predict cluster'):
    new_data = pd.DataFrame([[income, spending]], columns=['Annual Income (k$)', 'Spending Score (1-100)'])
    new_scaled = scaler.transform(new_data)
    cluster = kmeans.predict(new_scaled)[0]
    st.success(f"Predicted Cluster: {cluster} - {cluster_labels.get(cluster, 'Unknown')}")


    
