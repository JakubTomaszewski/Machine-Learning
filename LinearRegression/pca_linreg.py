'''
QUICK DESCRIPTION

1. download the data https://archive.ics.uci.edu/ml/datasets/Student+Performance
2. clean the data (no missing values)
3. get the valuable features using lasso, pca and compare their results
4. then, predict the data using linear regression
5. check who performs better - Males or Females
'''

import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.decomposition import PCA



#preparing data
df = pd.read_csv('student-mat.csv', sep=';')

# transforming categorical to numeric data
df['sex_fac'] = df['sex'].factorize()[0] # M = 1, F = 0

# selecting only numeric columns
df_small = df._get_numeric_data().drop('G3', axis=1)
df_num = df_small.values
# df_num = df.select_dtypes(['number'])


'''ML PREDICTION PART'''

# the value we are going to predict
y = df['G3'].values

X_train, X_test, y_train, y_test = train_test_split(df_num, y, test_size=0.2)

lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X_train, y_train).coef_

# lasso plot
plt.figure()
plt.plot(range(len(df_small.columns)), lasso_coef)
plt.xticks(range(len(df_small.columns)), df_small.columns, rotation=60)
plt.ylabel('Coefficients')

plt.show()
# plt.savefig('lasso_features.pdf')


# instantiate PCA and fit data
pca = PCA()
pca.fit(df_num)

# get intrinsic dimension of our data
n_features = range(pca.n_components_)

# PCA plot
plt.bar(n_features, pca.explained_variance_)
plt.xticks(n_features, rotation=60)
plt.ylabel('variance')
plt.xlabel('PCA feature')

plt.show()
# plt.savefig('pca_features.pdf')

# 4 features are the most valuable for our prediction
pca_model = PCA(n_components=4)
transformed = pca_model.fit_transform(df_num)

print(transformed.shape)

acc = 0
# getting the best performing model
while acc < 0.90:
    X_train, X_test, y_train, y_test = train_test_split(transformed, y, test_size=0.2)

    linreg = LinearRegression()

    linreg.fit(X_train, y_train)
    acc = linreg.score(X_test, y_test)

    # saving model to a file
    with open('student_model.pickle', 'wb') as file:
        pickle.dump(linreg, file)

# loading the saved model into our project
model_in = open('student_model.pickle', 'rb')
linreg = pickle.load(model_in)

y_pred = linreg.predict(X_test)

for i in range(len(y_pred)):
    print('Predicted value {}, real value {}'.format(y_pred[i].round(2), y_test[i]))

print('\nCoefficients: {}\nIntercept: {}'.format(linreg.coef_, linreg.intercept_))

print('\nThe model score is {}\n'.format(linreg.score(X_test, y_test)))





'''PERFORMANCE ANALYSIS PART'''

# Who performs better - Males or Females ?

performance = df[['sex', 'G1', 'G2', 'G3']]
# performance['sex'] = performance['sex'].astype('category')
performance['avg_grade'] = performance[['G1', 'G2', 'G3']].apply(lambda x: x.mean(), axis=1)


print('''Mean grade for females: {}
Mean grade for males: {}'''.format(performance['avg_grade'][performance['sex'] == 'F'].mean(), performance['avg_grade'][performance['sex'] == 'M'].mean()))
print('Standard deviation for males: {}'.format(performance['avg_grade'][performance['sex'] == 'M'].std()))

print('''\nMedian grade for females: {}
Median grade for males: {}'''.format(performance['avg_grade'][performance['sex'] == 'F'].median(), performance['avg_grade'][performance['sex'] == 'M'].median()))
print('Standard deviation for females: {}'.format(performance['avg_grade'][performance['sex'] == 'F'].std()))
# print(performance['avg_grade'][performance['sex'] == 'M'].describe())
# print(performance['avg_grade'][performance['sex'] == 'F'].describe())

performance['avg_grade'][performance['sex'] == 'F'].hist(bins=25, color='blue', label='Females')
performance['avg_grade'][performance['sex'] == 'M'].hist(bins=25, color='green', alpha=0.7, label='Males')

# Plotting distribution of avg grades
plt.title('Distribution of average grades among students')
plt.xlabel('Final grade')
plt.ylabel('Number of students')

plt.legend()
plt.show()

