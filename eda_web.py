import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from pydantic_settings import BaseSettings
#web app ka title
st.markdown('''
# **Exploratry data analysis web application**
This app is developed by codanics youtube channel called **EDA app**
''')
#how to upload file from pc
with st.sidebar.header("upload your dataset(.csv)"):
    uploaded_file=st.sidebar.file_uploader("upload your file",type=['csv'])
    df=sns.load_dataset('titanic')
    st.sidebar.markdown("[example csv file](df)")
#profiling report for pandas
if uploaded_file is not None:
    @st.cache_data
    def load_csv():
        csv=pd.read_csv(uploaded_file)
        return csv
    df=load_csv()
    pr=ProfileReport(df,explorative=True,config_file=None)
    st.header('**input DF**')
    st.write(df)
    st.write('---')
    st.header('**profiling report with pandas**')
    st_profile_report(pr)
else:
    st.info('awaiting for csv file,upload kr bhi do ab ya kam nhi krana')
    if st.button('press to use example data'):
        #example dataset
        @st.cache_data
        def load_data():
            a=pd.DataFrame(np.random.rand(100,5),columns=['age','bnana','codanics','ducthland','ear'])
            return a
        df=load_data()
        pr=ProfileReport(df,explorative=True,config_file=None)
        st.header('**input DF**')
        st.write(df)
        st.write('---')
        st.header('**profiling report with pandas**')
        st_profile_report(pr)

