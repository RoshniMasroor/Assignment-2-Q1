
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# NEW YORK
# Importing the dataset
dataset = pd.read_csv(r'D:\ML\assignment 2\assignment 2\50_Startups.csv')
N = dataset.loc[dataset.State=='New York', :] 
new_y = N.iloc[:, -1].values
q =np.arange(17)
new_x = q.reshape(-1, 1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(new_x, new_y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(new_x_train, new_y_train)

# predicting and Visualising the linear results
Y_pred = regressor.predict(new_x_test)
plt.scatter(new_x_train, new_y_train, color = 'red')
plt.plot(new_x_train, regressor.predict(new_x_train), color = 'blue')
plt.title('Profit of startups in New York (Linear Regression)')
plt.xlabel('New York')
plt.ylabel('Profit')
plt.show()
# since data is not linear so we move towards polynomial regression
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
new_x_poly = poly_reg.fit_transform(new_x)
poly_reg.fit(new_x_poly, new_y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(new_x_poly, new_y)

# Visualising the Polynomial Regression results
plt.scatter(new_x, new_y, color = 'red')
plt.plot(new_x, lin_reg_2.predict(poly_reg.fit_transform(new_x)), color = 'blue')
plt.title('Profit of startups in New York (Polynomial Regression)')
plt.xlabel('New York')
plt.ylabel('Profit')
plt.show()

print('Profit of startups in NYC (30) is:')
print(regressor.predict([[30]]))
# CALIFORNIA  
C = dataset.loc[dataset.State=='California', :]
cali_y = C.iloc[:, -1].values
s =np.arange(17)
cali_x = s.reshape(-1, 1)

# Splitting the dataset into the Training set and Test set
cali_x_train, cali_x_test, cali_y_train, cali_y_test = train_test_split(cali_x, cali_y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
regressor2 = LinearRegression()
regressor2.fit(cali_x_train, cali_y_train)

# Predicting the Test set results
Y_pred2 = regressor2.predict(cali_x_test)

#visualising the dataset and linear regression
plt.scatter(cali_x_train, cali_y_train, color = 'black')
plt.plot(cali_x_train, regressor.predict(cali_x_train), color = 'pink')
plt.title('Profit of startups in California (Linear Regression)')
plt.xlabel('California')
plt.ylabel('Profit')
plt.show()

# Fitting Polynomial Regression to the dataset
poly_reg2 = PolynomialFeatures(degree = 3)
cali_x_poly = poly_reg2.fit_transform(cali_x)
poly_reg.fit(cali_x_poly, cali_y)

lin_reg_3 = LinearRegression()
lin_reg_3.fit(cali_x_poly, cali_y)

# Visualising the Polynomial Regression results
plt.scatter(cali_x, cali_y, color = 'black')
plt.plot(cali_x, lin_reg_3.predict(poly_reg2.fit_transform(cali_x)), color = 'pink')
plt.title('Profit of startups in California (Polynomial Regression)')
plt.xlabel('California')
plt.ylabel('Profit')
plt.show()

print('Profit of startups in California (30) is:')
print(regressor2.predict([[30]]))

if regressor2.predict([[30]]) > regressor.predict([[30]]):
 { print('California earns more profit') }
else: print('Newyork earns more profit')
