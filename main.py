# library

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
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
- if we split the data in two buckets, since sensory data is not exact science. One of the >= 6, and the other less than 6,
 then %-wise we get 54% vs 46%
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
train_df_long = pd.melt(train_df,id_vars='sepal.length')
plt.figure(figsize=(16, 6))
sns.lmplot(x ='', y ='value', data = df,col='variable',
           col_wrap=2,aspect = 0.6, height,= 4, palette ='coolwarm', logistic= True)
plt.show()
