

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

class WineQualityClass:

    """
    TODO add broad description of the class and the functions and assumptions of all the files
    """
    def __init__(self, data_type_csv, data_type_csv_separator=';',logname = 'wine_quality.log'):
        self.data_types = pd.read_csv(data_type_csv, sep=data_type_csv_separator)
        self.logging.basicConfig(filename=logname,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
        #logger = logging.getLogger('urbanGUI')
    def _get_data_csv(self, path, seperator=';'):
        data = pd.read_csv(path, sep=seperator)
        return(data)

    def _validate_data(self,col_name,data):
        #we can add th col_name in the class itself after that
        # check if all necessary columns are present
        necessary_columns = self.data_types[col_name]
        if len(set(necessary_columns).difference(set(data))) == 0:
            self.logging.info("All columns are present correctly")
        else:
            self.logging.info("NOT all columns are present correctly")

        # check that the data types are as expected
        downloaded_data_type = pd.DataFrame(data.dtypes)
        downloaded_data_type.reset_index(inplace=True)
        downloaded_data_type.columns = self.data_types.columns
        check_df = downloaded_data_type.merge(self.data_types,
                                              right_on=self.data_types.columns[0],
                                              left_on=self.data_types.columns[0],
                                              how='left')
        check_df['type check'] = check_df[self.data_types.columns[0]+'_x'] == check_df[self.data_types.columns[0]+'_y']
        if sum(check_df['type check']) == len(check_df):
            self.logging.info("All data is in the correct type")
        else:
            self.logging.info("NOT all data is in the correct type")
