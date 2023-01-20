import pytest
import pandas as pd
import numpy as np


class TitanicPreprocessingDataTest:
    def __init__(self):
        return

    @pytest.fixture(scope='function')
    def data_fill_na(self):
        input_data_df = pd.DataFrame(data={'int_col': [1, 2, 3, None, 4, 0],
                                           'int_no_null_col': [0, 1, 2, 3, 4, 5],
                                           'str_col': ['A', '', None, 'sds', 'TRsd', 'AT_Aa'],
                                           'str_no_null_col': ['1', 'R', 'y', 'Ui', '@sd', 'ty_72'],
                                           'float_col': [3.1, -1.1, 2, 0, None, 6.123, -7871],
                                           'float_no_null_col': [3.1, -1.1, 2, 0, 3.2, 6.123, -7871]})

        checks = [(input_data_df, 'int_col', 0, pd.Series([1, 2, 3, 0, 4, 0])),
                  (input_data_df, 'int_col', 5, pd.Series([1, 2, 3, 5, 4, 0])),
                  (input_data_df, 'int_col', -2.1, pd.Series([1.0, 2.0, 3.0, -2.1, 4.0, 0.0])),

                  (input_data_df, 'int_no_null_col', 0, pd.Series([0, 1, 2, 3, 4, 5])),
                  (input_data_df, 'int_no_null_col', 3, pd.Series([0, 1, 2, 3, 4, 5])),
                  (input_data_df, 'int_no_null_col', -2, pd.Series([0, 1, 2, 3, 4, 5])),

                  (input_data_df, 'str_col', 'E', pd.Series(['A', '', 'E', 'sds', 'TRsd', 'AT_Aa'])),
                  (input_data_df, 'str_col', 'a', pd.Series(['A', '', 'a', 'sds', 'TRsd', 'AT_Aa'])),
                  (input_data_df, 'str_col', '0', pd.Series(['A', '', '0', 'sds', 'TRsd', 'AT_Aa'])),

                  (input_data_df, 'str_no_null_col', 'E', pd.Series(['1', 'R', 'y', 'Ui', '@sd', 'ty_72'])),
                  (input_data_df, 'str_no_null_col', 'a', pd.Series(['1', 'R', 'y', 'Ui', '@sd', 'ty_72'])),
                  (input_data_df, 'str_no_null_col', '0', pd.Series(['1', 'R', 'y', 'Ui', '@sd', 'ty_72'])),

                  (input_data_df, 'float_col', 0, pd.Series([3.1, -1.1, 2, 0, 0, 6.123, -7871])),
                  (input_data_df, 'float_col', 3.1, pd.Series([3.1, -1.1, 2, 0, 3.1, 6.123, -7871])),
                  (input_data_df, 'float_col', -0.2, pd.Series([3.1, -1.1, 2, 0, -0.2, 6.123, -7871])),

                  (input_data_df, 'float_no_null_col', 0, pd.Series([3.1, -1.1, 2, 0, 3.2, 6.123, -7871])),
                  (input_data_df, 'float_no_null_col', 3.1, pd.Series([3.1, -1.1, 2, 0, 3.2, 6.123, -7871])),
                  (input_data_df, 'float_no_null_col', -0.2, pd.Series([3.1, -1.1, 2, 0, 3.2, 6.123, -7871]))]
        return checks

    @pytest.fixture(scope='function')
    def data_dummify_categorical(self):
        input_data_df = pd.DataFrame(data={'A': ['a', 'b', 'a', 'a', 'a'],
                                           'B': ['b', 'a', 'c', 'c', np.nan],
                                           'C': [1, np.nan, 2, 3, -7.2]})
        # the expected is returned as the tuple
        # (expected prefix column, unique values, row indices per values, unchanged column list)
        checks = [(input_data_df, 'A', 'dummy', '_', 1, ('dummy_', ['a', 'b'], [[0, 2, 3, 4], [1]], ['B', 'C'])),
                  (input_data_df, 'B', 'dummy', '_', 1, ('dummy_', ['a', 'b', 'c'], [[1], [0], [2, 3]], ['A', 'C'])),
                  (input_data_df, 'A', 'dummy', '', 1, ('dummy', ['a', 'b'], [[0, 2, 3, 4], [1]], ['B', 'C'])),
                  (input_data_df, 'A', '', '', 1, ('', ['a', 'b'], [[0, 2, 3, 4], [1]], ['B', 'C'])),
                  (input_data_df, 'B', '', '', 3, ('', ['a', 'b', 'c'], [[1], [0], [2, 3]], ['A', 'C'])),
                  (input_data_df, 'B', 'col', '@', -4, ('col@', ['a', 'b', 'c'], [[1], [0], [2, 3]], ['A', 'C']))]
        return checks

    @pytest.fixture(scope='function')
    def data_women_children_first_rule(self):
        input_data_df = pd.DataFrame(data={'gender': ['M', 'male', 'F', 'female', 'M', 'male', 'female'],
                                           'age': [5, 20, 23, 3, 41, 5, 80],
                                           'other': ['_sd', 'rt', 'R.T. 6qpoD', 'M. Smith', 'lds8', '5', 'rt'],
                                           })

        checks = [(input_data_df, 'age', 'gender', 'female', "new_col", 1, pd.Series([1, 0, 0, 1, 0, 1, 1])),
                  (input_data_df, 'age', 'gender', 'female', "new_col", 6, pd.Series([6, 0, 0, 6, 0, 6, 6])),
                  (input_data_df, 'age', 'gender', 'female', "new_col", -2, pd.Series([-2, 0, 0, -2, 0, -2, -2])),
                  (input_data_df, 'age', 'gender', 'F', "new_col", 1.1, pd.Series([1.1, 0, 1.1, 1.1, 0, 1.1, 0])),
                  (input_data_df, 'other', 'gender', 'female', "new_col", 1, pd.Series([0, 0, 0, 0, 0, 0, 0])),
                  (input_data_df, 'age', 'other', 'female', "new_col", 1, pd.Series([0, 0, 0, 0, 0, 0, 0])),
                  (input_data_df, 'age_2', 'gender', 'female', "new_col", 1, pd.Series([0, 0, 0, 0, 0, 0, 0]))
                  ]
        return checks

    @pytest.fixture(scope='function')
    def data_preprocess(self):
        pass


class ClassifierDataTest:
    def __init__(self):
        pass


class StorageInterfaceDataTest:
    def __init__(self):
        pass