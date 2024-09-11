import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
#make container
header=st.container()
data_set=st.container()
features=st.container()
model_training=st.container()
with header:
    st.title("kashti ki app")
    st.text("In the project we will work on kashti data")
with data_set:
    st.header("kashti dobe gy,hawww")
    st.text("we will work on titanic dataset")
    #import dataset
    df=sns.load_dataset('titanic')
    df=df.dropna()
    st.write(df.head(10))
    st.subheader("ara o kitna aadmi tha?")
    st.bar_chart(df['sex'].value_counts())
    #other plots
    st.subheader("class k hisab sa faraq")
    st.bar_chart(df['class'].value_counts())
    #bar plots
    st.bar_chart(df['age'].sample(10))
with features:
    st.header("These are our app features:")
    st.text("awen bahut saray feature add krty ha, asan hi ha")
    st.markdown('1. **Feature 1:** This will tell us pta ni')
    st.markdown('2. **Feature 2:** This will tell us pta ni')
with model_training:
    st.header("kashti walo ka kia bna?-model training")
    st.text("we will increase or decrease our parameter here")
    #making columns
    input, display=st.columns(2)
    #pahla column ma apka selection point ho
    max_depth=input.slider("how many people do you know?",min_value=10,max_value=100,value=20,step=5)
# n-estimator
n_estimators=input.selectbox("How many trees should be there in a RF?",options=[50,100,200,300,'no limit'])
#adding list of features
input.write(df.columns)
#input feature from user
input_features=input.text_input('which feature should be use?')
#machine leaarning model
model=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)
#yaha per hum aik condition lgay ga
if n_estimators=='no limit':
        model=RandomForestRegressor(max_depth=max_depth)
else:
        model=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)
X=df[[input_features]]
y=df[['fare']]
#fit our model
model.fit(X,y)
pred=model.predict(X)
#display metrices
display.subheader("mean absolute error of the model is: ")
display.write(mean_absolute_error(y,pred))
display.subheader("mean squared error of the model is: ")
display.write(mean_squared_error(y,pred))
display.subheader("r squared score of the model is: ")
display.write(r2_score(y,pred))