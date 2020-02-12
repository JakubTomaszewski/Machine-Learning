'''
get the valuable features using lasso
then, predict the data using linear regression
check who performs better - Males or Females
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#preparing data
df = pd.read_csv('student-mat.csv', sep=';')
X = df[['G1', 'G2', 'studytime', 'failures', 'absences']].values
y = df['G3'].values

#splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_test)

for i in range(len(y_pred)):
    print('Predicted value {}, real value {}'.format(y_pred[i].round(2), y_test[i]))

print('\nThe score is {}'.format(linreg.score(X_test, y_test)))

print('\nCoefficients: {}\nIntercept: {}'.format(linreg.coef_, linreg.intercept_))

# check who performs better - Males or Females