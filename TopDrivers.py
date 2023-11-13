from WineQuality import WineQualityClass
import seaborn as sns
import matplotlib.pyplot as plt
csv_path = 'wine+quality/winequality-red.csv'
dtypes_path ='data_types.csv'
# initiate class
wine=WineQualityClass(csv_path=csv_path, dtype_path=dtypes_path)
# read the data from a csv
wine.get_data_csv()
# create binary variable for quality
wine.binary_variable_creation()
df = wine.data


#%%
under_inspection  = ['quality' ,'alcohol', 'sulphates', 'volatile acidity' ]
df[under_inspection].groupby(by='quality').mean().to_csv('top_drivers_statistics.csv')

