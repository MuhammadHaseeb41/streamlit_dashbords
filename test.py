import streamlit as st
import seaborn as sns
st.header("this is brought to you by #babaAmmar")
st.text("kia apko maza a raha ha")
st.header("pta ni kia likhna ha")
df=sns.load_dataset('iris')
st.write(df[['species','sepal_length','petal_length']].head(10))
st.bar_chart(df['sepal_length'])
st.line_chart(df['sepal_length'])