# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, [2]].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
# SVR needs feature scaling no inbuilt support is present for SVR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
#y =np.squeeze(sc_y.fit_transform(y.reshape(-1, 1)))
y = np.squeeze(sc_y.fit_transform(np.array(y))) 
#Above np.array converts into array because fir_transform needs 2D array and np.squeeze makes it vector again
#Both reshape and np.array serve the same purpose

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') #Gaussian model you can check other options with ctrl+i
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(sc_X.transform([[6.5]])) #Here just scaling is sufficient no need to fit already it got fitted above
y_pred = sc_y.inverse_transform(y_pred) #Once you get predicted value now you need inverse transform from -1 to 1 range to actual val

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()