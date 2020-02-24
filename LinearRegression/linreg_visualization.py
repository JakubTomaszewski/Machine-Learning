import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#preparing data
df = pd.read_csv('student-mat.csv', sep=';')

#instantiate model
linreg = LinearRegression()

x_train, x_test, Y_train, Y_test = train_test_split(df['G1'].values, df['G3'].values, test_size=0.2)
x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)

linreg.fit(x_train, Y_train)

print('Score is {}'.format(linreg.score(x_test, Y_test)))

print('The coefficient is: {}\nand the intercept: {}'.format(linreg.coef_, linreg.intercept_))

#model visualization
import matplotlib.pyplot as plt

plt.scatter(x_test, Y_test, color='red', marker='+')
plt.plot(x_test, linreg.predict(x_test), color='blue')

plt.xlabel('G1', fontsize=15)
plt.ylabel('G2', fontsize=15)
plt.show()
