

# TODO update the packages added by installing them first and having a list of necessary packages to install

# def import_or_install(package):
#     try:
#         __import__(package)
#     except ImportError:
#         pip.main(['install', package])

# library
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class WineQualityClass:

    """
    TODO add broad description of the class and the functions and assumptions of all the files
    """
    def __init__(self
                 ,csv_path
                 , dtype_path
                 , csv_separator=';'
                 , dtype_separator=';'
                 ,logname = 'wine.log'
                 ):
        self.data_types = pd.read_csv(dtype_path, sep=dtype_separator)
        self.csv_path = csv_path
        self.csv_separator = csv_separator
        logging.basicConfig(filename=logname
                            , filemode='a'
                            , format='%(asctime)s %(levelname)s %(message)s'
                            , datefmt='%H:%M:%S'
                            , level=logging.DEBUG)
    def _get_data_csv(self):
        self.data = pd.read_csv(self.csv_path, sep=self.csv_separator)
        logging.info("Data is read from a csv file successfully")
        #return self.data

    def _validate_data(self,col_name='column name'):
        #we can add th col_name in the class itself after that
        # check if all necessary columns are present
        necessary_columns = self.data_types[col_name]
        if len(set(necessary_columns).difference(set(self.data))) == 0:
            logging.info("All columns are present correctly")
        else:
            logging.warning("NOT all columns are present correctly")

        # check that the data types are as expected
        downloaded_data_type = pd.DataFrame(self.data.dtypes)
        downloaded_data_type.reset_index(inplace=True)
        downloaded_data_type.columns = self.data_types.columns
        check_df = downloaded_data_type.merge(self.data_types,
                                              right_on=self.data_types.columns[0],
                                              left_on=self.data_types.columns[0],
                                              how='left')
        check_df['type check'] = check_df[self.data_types.columns[1]+'_x'] == check_df[self.data_types.columns[1]+'_y']
        if sum(check_df['type check']) == len(check_df):
            logging.info("All data is in the correct type")
        else:
            logging.warning("NOT all data is in the correct type")

    def _binary_variable_creation(self
                           , old_col_name = 'quality'
                           , new_col_name = 'quality bucket'
                           , operator = '>'
                           , criteria = 5
                           , label1 = 'high'
                           , label2 = 'low'
                           ):
        self.label1 = label1
        if operator == '>':
            try:
                self.data[new_col_name] = np.where(self.data[old_col_name] > criteria, label1, label2)
                logging.info("New variable created")
            except  (AttributeError, TypeError):
                logging.error("CANNOT create a new variable since the criteria is not an integer")
                raise AssertionError("CANNOT create a new variable since the criteria is not an integer")
        elif operator == '<':
            try:
                self.data[new_col_name] = np.where(self.data[old_col_name] < criteria, label1, label2)
                logging.info("New variable created")
            except  (AttributeError, TypeError):
                logging.error("CANNOT create a new variable since the criteria is not an integer")
                raise AssertionError("CANNOT create a new variable since the criteria is not an integer")
        elif operator == '==':
            try:
                self.data[new_col_name] = np.where(self.data[old_col_name] == criteria, label1, label2)
                logging.info("New variable created")
            except  (AttributeError, TypeError):
                logging.error("CANNOT create a new variable since the criteria is not an integer")
                raise AssertionError("CANNOT create a new variable since the criteria is not an integer")
        else:
            logging.error("CANNOT create a new variable since the operator is not supported")
            raise AssertionError("CANNOT create a new variable since the operator is not supported")


    def _data_split(self
                    ,y_column = 'quality bucket'
                    ,stratify_column = 'quality' # TODO to handle if this is null or None
                    ,size = 0.25
                    ):
        self.y_column = y_column
        self.y_columns = [stratify_column, self.y_column]
        all_columns = self.data.columns
        self.x_columns = all_columns[~all_columns.isin(self.y_columns)]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.loc[:, self.x_columns]
                                                            ,self.data[self.y_column]
                                                            ,stratify=self.data[stratify_column]
                                                            ,test_size=size)

        # save files for future use
        self.X_train.to_csv('X_train.csv', index=False)
        self.X_test.to_csv('X_test.csv', index=False)
        self.y_train.to_csv('y_train.csv', index=False)
        self.y_test.to_csv('y_test.csv', index=False)
        logging.info("Train and test split of the data")

        #return X_train, X_test, y_train, y_test

    def _plot_figure(self
                     , plot_title
                     , file_title
                     , plot_code):
        plt.figure(figsize=(16, 6))
        exec(plot_code)
        plt.title(plot_title)
        plt.savefig(file_title, dpi=300, bbox_inches='tight')

    def _correlation_analysis(self):
        self.corr = self.X_train.corr()
        self._plot_figure(
            plot_title= 'Feature Correlation'
            , file_title = 'feature_correlation.png'
            , plot_code= 'sns.heatmap(self.corr, annot=True)'
        )

        #TODO assert that the inputs are strings and make them dynamic

    def _pca_analysis(self, n = 2):
        pca = PCA(n_components=n)
        pca.fit(self.X_train)
        return pca.explained_variance_ratio_


    def _independent_logistic_regression(self):
        train_df = pd.concat([self.X_train, self.y_train], axis=1)
        self.melt_df = pd.melt(train_df, id_vars=[self.y_column]
                          , value_vars=self.x_columns)
        self.melt_df[self.y_column] = np.where(self.melt_df[self.y_column] == self.label1, 1, 0)
        code_str = "sns.lmplot(x='value', y=self.y_column, data=self.melt_df, col='variable', col_wrap=2, aspect=0.6, height=4,palette='coolwarm', logistic=True)"
        self._plot_figure(
            plot_title='Logistic Regressions'
            ,file_title='logistic_regression.png'
            ,plot_code=code_str
        )

    # TODO the title of the graph is not visible and also to add the inputs of the graph into the func itself to make it dynamic

    def _best_model(self,model,param_dist):
        best_model = RandomizedSearchCV(model,
                                         param_distributions=param_dist,
                                         n_iter=5,
                                         cv=5)
        return best_model

    def _random_forest(self):
        param_dist = {'n_estimators': randint(50, 500),
                      'max_depth': randint(1, 20)}



