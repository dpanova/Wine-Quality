# library
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

class WineQualityClass:
    """WineQuality class automates a report for wine quality.

    :param csv_path: indicates path to the input data
    :type csv_path: str
    :param dtype_path: indicates path to the file which contains the necessary data types and columns in order for the analysis to be conducted
    :type dtype_path: str
    :param csv_separator: indicates the separator for the csv_path file. The default value is ';'
    :type csv_separator: str
    :param dtype_separator: indicates the separator for the dtype_csv file. The default value is ';'
    :type dtype_separator: str
    :param logname: indicates path to the log file, The default value is 'wine.log'
    :type logname: str
    :return: WineQualityClass object
    :rtype: WineQualityClass
    """
    def __init__(self
                 , csv_path
                 , dtype_path
                 , csv_separator=';'
                 , dtype_separator=';'
                 , logname='wine.log'
                 ):

        self.data_types = pd.read_csv(dtype_path, sep=dtype_separator)
        self.csv_path = csv_path
        self.csv_separator = csv_separator
        logging.basicConfig(filename=logname
                            , filemode='a'
                            , format='%(asctime)s %(levelname)s %(message)s'
                            , datefmt='%H:%M:%S'
                            , level=logging.DEBUG)
        # TODO add as a return values the self objects

    def get_data_csv(self):
        """Goal: The function takes the csv_path location from the class and the corresponding csv_separator to the read the data

        :return: pandas data frame with the results
        """
        self.data = pd.read_csv(self.csv_path, sep=self.csv_separator)
        logging.info("Data is read from a csv file successfully")

    def validate_data(self, col_name='column name'):
        """
        :param col_name: string of the column name which indicates where in the dtype_csv file the necessary column names are
        :return: validation if the read csv is as expected so that the analysis can be conducted
        """
        # we can add th col_name in the class itself after that
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
        check_df['type check'] = check_df[self.data_types.columns[1] + '_x'] == check_df[
            self.data_types.columns[1] + '_y']
        if sum(check_df['type check']) == len(check_df):
            logging.info("All data is in the correct type")
        else:
            logging.warning("NOT all data is in the correct type")

    def binary_variable_creation(self
                                 , old_col_name='quality'
                                 , new_col_name='quality bucket'
                                 , operator='>'
                                 , criteria=5
                                 , label1='high'
                                 , label2='low'
                                 , distribution_plot=True
                                 ):
        """
        Goal: This function can be used to create a new binary variable from an ordered categorical variable
        :param old_col_name: string indicating the ordered categorical variable column name. The default value is 'quality'
        :param new_col_name: string indicating the new categorical variable column name. The default value is 'quality bucket'
        :param operator: string which indicated what the operator for the creation of the new variable should be. The options are between ('>','<','=='). The default is '>'
        :param criteria: integer indicating the ordered categorical variable cut-off point for the creation of the new variable.The default is 5
        :param label1: string indicating the one of the labels for the new variable.The default is 'high'
        :param label2: string indicating the other label for the new variable. The default value is 'low'
        :param distribution_plot: binary variable indicating if distribution graphs should be drawn and saved
        :return: a new column in the dataframe and graphs of the dependant variables
        """
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

        if distribution_plot:
            # old column distribution
            self.old_col_name = old_col_name
            self.old_df = pd.DataFrame(self.data[self.old_col_name].value_counts())
            self.old_df.reset_index(inplace=True)
            self.old_df.sort_values(ascending=False, by=self.old_col_name, inplace=True)

            code_str = "sns.barplot(self.old_df, x=self.old_col_name, y='count')"

            self.plot_figure(
                plot_title=self.old_col_name + ' distribution'
                , file_title='dependant_original.png'
                , plot_code=code_str
            )

            logging.info("Old variable distribution plot created")

            # new column distribution
            self.new_col_name = new_col_name
            self.new_df = pd.DataFrame(self.data[self.new_col_name].value_counts())
            self.new_df.reset_index(inplace=True)
            self.new_df.sort_values(ascending=False, by=self.new_col_name, inplace=True)

            code_str = "sns.barplot(self.new_df, x=self.new_col_name, y='count')"

            self.plot_figure(
                plot_title=self.new_col_name + ' distribution'
                , file_title='dependant_created.png'
                , plot_code=code_str
            )
            logging.info("New variable distribution plot created")

    def data_split(self
                   , y_column='quality bucket'
                   , stratify_column='quality'
                   , size=0.25
                   ):
        """

        :param y_column: string indicating the column name of the dependant variable. The default value is 'quality bucket'.
        :param stratify_column: sting indicating the column name of the variable to do a stratified sample on. The default value is 'quality'.
        :param size: float indicating the test size
        :return: X_train, y_train, X_test and y_test data frames as well as saving them as csv files with the same naming conventions
        """
        try:
            self.y_column = y_column
            self.y_columns = [stratify_column, self.y_column]
            all_columns = self.data.columns
            self.x_columns = all_columns[~all_columns.isin(self.y_columns)]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.loc[:, self.x_columns]
                                                                                    , self.data[self.y_column]
                                                                                    ,
                                                                                    stratify=self.data[stratify_column]
                                                                                    , test_size=size)

            # save files for future use
            self.X_train.to_csv('X_train.csv', index=False)
            self.X_test.to_csv('X_test.csv', index=False)
            self.y_train.to_csv('y_train.csv', index=False)
            self.y_test.to_csv('y_test.csv', index=False)
            logging.info("Train and test split of the data")
        except Exception as error:
            logging.error(error)
        # return X_train, X_test, y_train, y_test

    def plot_figure(self
                    , plot_title
                    , file_title
                    , plot_code):
        """
        :param plot_title: string indicating the plot title
        :param file_title: string indicating the title of the saved file
        :param plot_code: string with the code to be executed for the plot to visualize
        :return: saved file with the requested plot
        """
        try:
            plt.figure(figsize=(16, 6))
            exec(plot_code)
            plt.title(plot_title)
            plt.savefig(file_title, dpi=300, bbox_inches='tight')
            logging.info("Graph is plotted")
        except Exception as error:
            logging.error(error)

    def correlation_analysis(self):
        """
        The Goal of this analysis is to check if the dependent variables are correlated
        :return: 'feature_correlation.png' - a plot with the correlation matrix
        """
        try:
            self.corr = self.X_train.corr()
            self.plot_figure(
                plot_title='Feature Correlation'
                , file_title='feature_correlation.png'
                , plot_code='sns.heatmap(self.corr, annot=True)'
            )
            logging.info("Correlation analysis completed")
        except Exception as error:
            logging.error(error)

    def pca_analysis(self, n=2):
        """
        Goal: the higher the explained variance is, the higher the correlation is
        :param n: number of components for the PCA
        :return: a list with the explained variance from the PCA components
        """
        try:
            pca = PCA(n_components=n)
            pca.fit(self.X_train)
            logging.info('PCA Analysis completed')
        except Exception as error:
            logging.error(error)
        return pca.explained_variance_ratio_

    def independent_logistic_regression(self):
        """
        Goal: Run a logistic regression against each dependant variable separately to identify linear relationship
        :return: 'logistic_regressions.png' - a plot with the logistic graphs.
        """
        try:
            train_df = pd.concat([self.X_train, self.y_train], axis=1)
            self.melt_df = pd.melt(train_df, id_vars=[self.y_column]
                                   , value_vars=self.x_columns)
            self.melt_df[self.y_column] = np.where(self.melt_df[self.y_column] == self.label1, 1, 0)
            code_str = "sns.lmplot(x='value', y=self.y_column, data=self.melt_df, col='variable', col_wrap=4, aspect=0.6, height=4,palette='coolwarm', logistic=True)"
            self.plot_figure(
                plot_title=''
                , file_title='logistic_regression.png'
                , plot_code=code_str
            )
            logging.info('Logistic regressions against all independent variables completed')
        except Exception as error:
            logging.error(error)

    def best_model(self, model, param_dist):
        """
        Goal: The goal of the function is to identify the best model after tuning the hyperparameters
        :param model: an initiated machine learning model
        :param param_dist: a dictionary with the parameters and their respective ranges for the tuning
        :return: a RandomizedSearchCV object
        """
        try:
            best_model = RandomizedSearchCV(model,
                                            param_distributions=param_dist,
                                            n_iter=100,
                                            cv=10)
            logging.info('Parameter tuning completed')
        except Exception as error:
            logging.error(error)
        return best_model

    def accuracy_metrics(self, y_pred, cm_title, cm_file_name):
        """
        Goal: The goal of this function is to provide accuracy metrics to compare the different models
        :param y_pred: list of predicted dependent values
        :param cm_title: string identifying the title for the confusion matrix plot
        :param cm_file_name: string identifying the file title for the confusion matrix plot
        :return: accuracy, precision, recall and saved grpah for the confusion matrix
        """
        try:
            # accuracy score
            acc = accuracy_score(self.y_test, y_pred)
            # calculate the precision score
            precision = precision_score(self.y_test, y_pred, average='macro')
            recall = recall_score(self.y_test, y_pred, average='macro')
            # confusion matrix

            self.cm = confusion_matrix(self.y_test, y_pred)

            # code_str = """
            # sns.heatmap(self.cm, annot=True, fmt='.3f', linewidths=.5, square=True, cmap='Blues_r');
            # plt.ylabel('Actual label');
            # plt.xlabel('Predicted label');
            #             """

            code_str = "sns.heatmap(self.cm, annot=True, fmt='.3f', linewidths=.5, square=True, cmap='Blues_r')"

            self.plot_figure(
                plot_title=cm_title
                , file_title=cm_file_name
                , plot_code=code_str
            )
            logging.info('Accuracy metrics calculated')
        except Exception as error:
            logging.error(error)

        return acc, precision, recall

    def misclassified_analysis(self, y_pred):
        """
        Goal: the goal of this analysis is to understand where the model miscalculates and if any pattern can be found
        :param y_pred: list of predicted dependent values
        :return: list of the misclassified values
        """
        try:

            # check the misclassified datapoints
            y_test_df = pd.DataFrame(self.y_test)
            y_test_df['pred'] = y_pred
            y_test_df['check'] = y_test_df[self.y_column] == y_test_df['pred']

            original_y_column_list = [y for y in self.y_columns if y != self.y_column]
            if len(original_y_column_list) == 0:
                original_y_column = self.y_column
            else:
                original_y_column = original_y_column_list[0]
            misclassified = self.data.loc[y_test_df[~y_test_df['check']].index, original_y_column].value_counts()

            logging.info('Misclassified analysis completed')
        except Exception as error:
            logging.error(error)

        return misclassified

    def model_results(self
                      , model
                      , param_dist
                      , cm_title
                      , cm_file_name):
        """
        Goal: This function is to provide a full picture of the model performance and accuracy
        :param model: an initiated machine learning model
        :param param_dist: a dictionary with the parameters and their respective ranges for the tuning
        :param cm_title: string identifying the title for the confusion matrix plot
        :param cm_file_name: string identifying the file title for the confusion matrix plot
        :return: the best model with its accuracy metrics  and misclassified analysis
        """
        try:
            # Use random search to find the best hyperparameters
            rand_search = self.best_model(model=model, param_dist=param_dist)

            # fit the best model
            rand_search.fit(self.X_train, self.y_train)

            # generate predictions with the best model
            y_pred = rand_search.predict(self.X_test)

            # calculate model accuracy
            acc, precision, recall = self.accuracy_metrics(y_pred=y_pred,
                                                           cm_title=cm_title,
                                                           cm_file_name=cm_file_name)

            # check the misclassified datapoints
            misclassified = self.misclassified_analysis(y_pred=y_pred)

            logging.info('Model Results Calculated')
        except Exception as error:
            logging.error(error)

        return acc, precision, recall, misclassified, rand_search

    def random_forest_results(self):
        """
        Goal: Run Random Forest and conduct hyperparameter tuning, accuracy measurement and feature importance
        :return: accuracy, precision, recall, confusion matrix plot file with the name 'rf_confusion_matrix.png',
        misclassified analysis, feature importance plot with the name 'rf_feature_importance.png'
        """
        try:
            param_dist = {'n_estimators': randint(50, 500), 'max_depth': randint(1, 20)}
            # Create a random forest classifier
            rf = RandomForestClassifier()

            # check the model results
            acc, precision, recall, misclassified, rand_search = self.model_results(model=rf, param_dist=param_dist,
                                                                                    cm_title='RF Confusion Matrix',
                                                                                    cm_file_name='rf_confusion_matrix.png')

            # # check the features importance
            best_model = rand_search.best_estimator_
            importances = best_model.feature_importances_
            self.forest_importances = pd.Series(importances, index=self.X_train.columns)

            code_str = """ 
            self.forest_importances.sort_values(ascending=False).plot(kind='barh')
            plt.ylabel('Importance')
            plt.xlabel('Features')
                        """

            self.plot_figure(
                plot_title='RF Feature Importance'
                , file_title='rf_feature_importance.png'
                , plot_code=code_str
            )
            logging.info('Random forest results calculated')

            # create a plot for the misclassified
            self.misclass_rf = pd.DataFrame(misclassified)
            self.misclass_rf.reset_index(inplace=True)
            self.misclass_rf.sort_values(ascending=False, by=self.old_col_name, inplace=True)

            code_str = "sns.barplot(self.misclass_rf, x=self.old_col_name, y='count')"

            self.plot_figure(
                plot_title='Random Forest Misclassified Distribution'
                , file_title='misclassified_rf.png'
                , plot_code=code_str
            )
            logging.info("Random Forest misclassified distribution plot created")

            return acc, precision, recall, misclassified
        except Exception as error:
            logging.error(error)

    def log_reg_results(self):
        """
        Goal: Run Logistic Regression and conduct hyperparameter tuning, accuracy measurement and feature importance
        :return: accuracy, precision, recall, confusion matrix plot file with the name 'rf_confusion_matrix.png',
        misclassified analysis, feature importance plot with the name 'lasso_confusion_matrix.png'
        """
        try:
            param_dist = {'C': np.arange(0.01, 0.5, 0.01)}

            # create a logistic regression model
            log_reg = LogisticRegression(
                random_state=42,
                penalty='l1',
                solver='liblinear'
            )

            # check the model results
            acc, precision, recall, misclassified, rand_search = self.model_results(model=log_reg
                                                                                    , param_dist=param_dist
                                                                                    ,
                                                                                    cm_title='Lasso Confusion Matrix',
                                                                                    cm_file_name='lasso_confusion_matrix.png')

            # check the features importance
            best_model = rand_search.best_estimator_
            log_reg_coef = list(zip(best_model.coef_[0], self.X_train.columns))
            log_reg_coef_non_zero = [item for item in log_reg_coef if item[0] != 0]
            logging.info('Logistic regression results calculated')

            # create a plot for the misclassified
            self.misclass_log = pd.DataFrame(misclassified)
            self.misclass_log.reset_index(inplace=True)
            self.misclass_log.sort_values(ascending=False, by=self.old_col_name, inplace=True)

            code_str = "sns.barplot(self.misclass_log, x=self.old_col_name, y='count')"

            self.plot_figure(
                plot_title='Lasso Misclassified Distribution'
                , file_title='misclassified_log.png'
                , plot_code=code_str
            )
            logging.info("Lasso misclassified distribution plot created")

            return acc, precision, recall, misclassified, log_reg_coef_non_zero
        except Exception as error:
            logging.error(error)
