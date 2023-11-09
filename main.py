# library

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
#%%
data = pd.read_csv('wine+quality/winequality-red.csv', sep=';') # need to write the expectation of the seperator
data_types = pd.read_csv('data_types.csv', sep=';')

# check if all necessary columns are present
necessary_columns = data_types['column name']

if len(set(necessary_columns).difference(set(data))) == 0:
    print('OK columns')
else:
    print('Not OK columns')

# TODO log and error handling

# check that the data types are the expected ones
downloaded_data_type = pd.DataFrame(data.dtypes)
downloaded_data_type.reset_index(inplace=True)
downloaded_data_type.columns = ['column name', 'column type']
check_df = downloaded_data_type.merge(data_types, right_on = 'column name', left_on ='column name',how='left')
check_df['type check'] = check_df['column type_x'] ==check_df['column type_y']
if sum(check_df['type check']) == len(check_df):
    print('same data type')
else:
    print('different data type')
# TODO assert no null values in the dataset

#%%
# let us see the distribution of the dependant variable
data['quality'].value_counts()

"""
Observations: 
- there are missing qualities 
- the lowest is 3 and the highest is 8 
- majority of the data points are above 5
- if we split the data in two buckets, since sensory data is not exact science. 
One of the >= 6, and the other less than 6,  then %-wise we get 54% vs 46%
- we can call the new buckets low and high quality 
"""

# adding the new buckets
data['quality bucket'] = np.where(data['quality']>5, 'high','low')
data['quality bucket'].value_counts()

# let us split the data to train and test
y_columns = ['quality','quality bucket']
X_train, X_test, y_train, y_test = train_test_split(data.loc[:,~data.columns.isin(y_columns)],
                                                    data['quality bucket'],
                                                    test_size=0.25)

# save files for future use
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv',index=False)
y_test.to_csv('y_test.csv',index=False)

#%%
# now let us check the correlation between the different features
corr = X_train.corr()
plt.figure(figsize=(16, 6))
sns.heatmap(corr, annot=True)
plt.title('Feature Correlation')
#plt.show()
plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight')

"""
We use Pearson correlation. It looks like acidity and pH are highly correlated (above 0.5). 
Additionally, density and alcohol as well. We can observe 0.3 as well for other pairs. 
we need to use regularization techniques to ensure this is not affecting the results. 
"""

#%%
# let us do correlation analysis on the dependant and independent variables to see if we see any correlation, not that
# correlation means causation

train_df = pd.concat([X_train, y_train], axis=1)
melt_df = pd.melt(train_df, id_vars=['quality bucket'], value_vars=data.columns[~data.columns.isin(y_columns)])
melt_df['quality bucket'] = np.where(melt_df['quality bucket']=='high', 1,0)

plt.figure(figsize=(16, 6))
sns.lmplot(x ='value', y ='quality bucket', data = melt_df,col='variable', col_wrap=2,aspect = 0.6, height= 4,
           palette ='coolwarm', logistic = True)
plt.savefig('logistic_regression.png', dpi=300, bbox_inches='tight')
#plt.show()

"""
From the individual-logistic-regressions graph, it looks like free sulfur dioxide, total suffer dioxide 
(higher than the first one), residual sugar and fixed acidity have some variance against the quality bucket     
"""

#%%
#let us continue with models
# we will start with Logistic regression with regularization
"""
Here we will use logistic regression to with regularization:
- L1 (Lasso) encourages the model to have sparse coefficients, which leads to most of the coefficients to have a value, which is 
actually feature engineering. 
Recursive Feature Elimination (RFE): RFE is a feature selection technique that involves recursively removing the least 
important features from the model until the desired number of features is reached. 
It can be used in combination with any linear model that has a regularization term, such as Lasso or Ridge Regression.

Here we will start with Lasso and after that apply RFE, after all the goal is identify the drivers. 

L1 results: 
- we have 0.7175 accuracy, which is satisfactory to some extend
- precision is similar 0.715 -> predicting correctly the true sign 
- slightly worse recall score 0.710

Recommended features to look into for the high quality red wine based on L1 log regression are ['fixed acidity', 'volatile acidity', 'free sulfur dioxide', 'total sulfur dioxide', 'pH', 'sulphates', 'alcohol']
We should note that acids and pH are highly correlated as per the correlation analysis. Additionally, both dioxides are in as expected by the single log regressions 

As expected, majority of the misclassified cases are between 5 and 6 quality.

The score is similar with FRE, slightly better 0.71 

Q: can we say that we don't have linear relation based on the results?
"""

# TODO provide justification of the parameters from here https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
log_reg = LogisticRegression(C = 0.1, penalty='l1', random_state = 42,solver= 'liblinear')
log_reg.fit(X_train, y_train)
score = log_reg.score(X_test, y_test) #0.7175
# add the confusion matrix
predictions = log_reg.predict(X_test)
cm = metrics.confusion_matrix(y_test, predictions)

# plot the confusion matrix
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);
plt.show()

# calculate the precision score
precision_score(y_test, predictions, average='macro')
recall_score(y_test, predictions, average='macro')


# get the coefficients to identify the drivers of the high quality wine
log_reg_coef = list(zip(log_reg.coef_[0], X_train.columns))
log_reg_coef_non_zero = [item[1] for item in log_reg_coef if item[0]!=0]


# then we need to figure out if the misclassified results are just two close to each other in quality

y_test_df = pd.DataFrame(y_test)
y_test_df['pred'] = predictions
y_test_df['check'] = y_test_df['quality bucket'] == y_test_df['pred']

data.loc[y_test_df[~y_test_df['check']].index,'quality'].value_counts()


# rfe

log_reg = LogisticRegression(C = 0.1, penalty='l1', random_state = 42,solver= 'liblinear')
selector = RFE(log_reg)
selector.fit(X_train, y_train)
score = selector.score(X_test, y_test) #0.7175
# add the confusion matrix
predictions = selector.predict(X_test)
cm = metrics.confusion_matrix(y_test, predictions)

#%%
# let us fit Random Forest model
"""
Do we actually need feature selection for Random Forest - if I remember correctly, no 
We can do a graph for the leaves 
It looks like those are better results than the log regression 
0.7975 is the accuracy, precision and recall are similar as well as in the log regression 
Same for the misclassified - 5 and 6 are mostly misclassified 
"""
param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf,
                                 param_distributions = param_dist,
                                 n_iter=5, # we can change that
                                 cv=5) #we can change that

# Fit the random search object to the data
rand_search.fit(X_train, y_train)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

print('Best hyperparameters:',  rand_search.best_params_)

# Generate predictions with the best model
y_pred = best_rf.predict(X_test)

# score
accuracy_score(y_test, y_pred)

# Create the confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(confusion_matrix=cm).plot() #this looks like a better plot than the previous one

# we can check which are the misclassified qualities

y_test_df = pd.DataFrame(y_test)
y_test_df['pred'] = y_pred
y_test_df['check'] = y_test_df['quality bucket'] == y_test_df['pred']

data.loc[y_test_df[~y_test_df['check']].index,'quality'].value_counts()

# check the features importance
importances = best_rf.feature_importances_
forest_importances = pd.Series(importances, index=X_train.columns)


# to update
fig, ax = plt.subplots()

ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")

