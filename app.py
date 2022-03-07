
import streamlit as st
st.title("Iris Classifier- API")
sl = st.slider('Sepal Length', 4.3, 7.9, 1)
sw = st.slider('Sepal Width', 2.0, 4.4, 1)
pl = st.slider('Petal Length', 1.0, 6.9, 1)
pw = st.slider('Petal Width', 0.1, 2.5, 1)

from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data
y = iris.target

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x,y)

op = model.predict([[sl,sw,pl,pw]])
op = iris.target_names[op[0]]
st.title(op)   



