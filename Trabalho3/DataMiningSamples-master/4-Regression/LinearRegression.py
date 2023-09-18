import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
#from sklearn.datasets import fetch_openml

# carrega os dados
house_data = fetch_california_housing()
#house_data = fetch_openml(name="house_prices", as_frame=True)
X = house_data['data']
y = house_data['target']
df = pd.DataFrame(data=house_data.data, columns=house_data.feature_names)
print(df.head())

# separa em set de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

regr = LinearRegression()
regr.fit(X_train, y_train)

r2_train = regr.score(X_train, y_train)
r2_test = regr.score(X_test, y_test)
print('R2 no set de treino: %.2f' % r2_train)
print('R2 no set de teste: %.2f' % r2_test)

y_pred = regr.predict(X_test)
abs_error = mean_absolute_error(y_pred, y_test)
print('Erro absoluto no set de treino: %.2f' % abs_error)