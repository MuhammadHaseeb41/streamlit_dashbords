import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#app ki heading
st.write(
    """
# explore different ml models and datasets
dakhta ha konsa best ha en ma sa? 
"""
)
#datasets ka name aik box ma dal k sidebar pa laga do
dataset_name=st.sidebar.selectbox(
    'select dataset',
    ('Iris','Breast Cancer','wine')
)
# or esi k neecha classifier k name aik dba ma dal do
classifier_name=st.sidebar.selectbox('select classifier',('SVM','KNN','Random Forest'))
# ab hum na aik function define krna ha dataset ko load krna k lia
def get_dataset(dataset_name):
    data=None
    if dataset_name=='Iris':
        data=datasets.load_iris()
    elif dataset_name=='wine':
        data=datasets.load_wine()
    else:
        data=datasets.load_breast_cancer()
    x=data.data
    y=data.target
    return x,y
#ab es function ko bula la ga aor x,y variable k equal rukh da ga
X,y=get_dataset(dataset_name)
# ab hum apna data set ki shape ko app pa print kr da ga
st.write('shape of dataset',X.shape)
st.write('number of classes',len(np.unique(y)))
#next hum different cllasifier ka parameter ko user input ma add kra ga
def add_parameter_ui(classifier_name):
    params=dict() #create an empty dictionary
    if classifier_name=='SVM':
        C=st.sidebar.slider('C',0.01,10.0)
        params['C']=C #its the degree of correct classification
    elif classifier_name=='KNN':
        K=st.sidebar.slider('K',1,15)
        params['K']=K #its the number of nearest neighbour
    else:
        max_depth=st.sidebar.slider('max_depth',2,15)
        params['max_depth']=max_depth # depth of every tree which grow in random forest
        n_estimators=st.sidebar.slider('n_estimator',1,100)
        params['n_estimators']=n_estimators #number of tree in random forest
    return params
#ab es function ko bula la ga aor param variable k equal rukh da
params=add_parameter_ui(classifier_name)
# ab hum classifier bnay ga base on classifier_name and params
def get_classifier(classifier_name,params):
    clf=None
    if classifier_name=='SVM':
        clf=SVC(C=params['C'])
    elif classifier_name=='KNN':
        clf=KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf=clf=RandomForestClassifier(n_estimators=params['n_estimators'],max_depth=params['max_depth'],
                                       random_state=1234)
    return clf
#ab es function ko bula la ga aor clf k equal rkh da ga
clf=get_classifier(classifier_name,params)
#ab hum apna dataset ko test aor train ma split kr deta ha by 80/20 ratio
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)
#ab hum na classifier ko train krwa dena ha apna data per
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
#model ka accuracy score check kr la ga aor usa app pa print bhi krwa da ga
acc=accuracy_score(y_test,y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =',acc)
#plot dataset
#ab hum apnay sara feature ko 2 dimensional plot pa dra kr da ga using pca
pca=PCA(2)
X_projected=pca.fit_transform(X)
#ab hum apna data 0 aor 1 dimension ma slice kr k da ga
x1=X_projected[:,0]
x2=X_projected[:,1]
fig=plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap='viridis')
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.colorbar()
#plt.show
st.pyplot(fig)