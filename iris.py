import pandas as pd
from knn import *
from utils import *

data = pd.read_csv("../data/iris.csv")

x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

x_train, y_train, x_test, y_test = test_train_split(x,y,.3)

model = KNN(5)
model.fit(x_train, y_train)

preds = model.predict(x_test)

# Accuracy
print(f"Accuracy is {(preds == y_test).mean()}")
