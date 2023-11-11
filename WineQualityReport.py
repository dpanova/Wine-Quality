from WineQuality import WineQualityClass
csv_path = 'wine+quality/winequality-red.csv'
dtypes_path ='data_types.csv'

wine=WineQualityClass(csv_path=csv_path, dtype_path=dtypes_path)

wine._get_data_csv()
wine._validate_data()
wine._binary_variable_creation()
wine._data_split()
wine._correlation_analysis()
wine._pca_analysis()
wine._independent_logistic_regression()


