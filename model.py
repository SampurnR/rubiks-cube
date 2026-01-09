import pandas as pd
import sklearn

from sklearn.datasets import load_iris
data = load_iris()
X = data.features
y = data.target

from sklearn.linear import LogisticRegression

model = LogisticRegression()
model.fit(X,y)

print(model.predict(X))
