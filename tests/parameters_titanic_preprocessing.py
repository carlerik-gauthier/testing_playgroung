"""
Section "For testcases that are used by multiple test functions" from
Testing Best Practices for Machine Learning Libraries
 https://towardsdatascience.com/testing-best-practices-for-machine-learning-libraries-41b7d0362c95

might have been another solution
"""

import pandas as pd
import numpy as np
from copy import deepcopy
# from sklearn.ensemble import RandomForestClassifier


class TitanicPreprocessingDataTest(object):
    def __init__(self):
        return

    @staticmethod
    def data_fill_na():
        input_data_df = pd.DataFrame(data={'int_col': [1, 2, 3, None, 4, 0, None],
                                           'int_no_null_col': [0, 1, 2, 3, 4, 5, 6],
                                           'str_col': ['A', '', None, 'sds', 'TRsd', 'AT_Aa', ' '],
                                           'str_no_null_col': ['1', 'R', 'y', 'Ui', '@sd', 'ty_72', 'None'],
                                           'float_col': [3.1, -1.1, 2, 0, None, 6.123, -7871],
                                           'float_no_null_col': [3.1, -1.1, 2, 0, 3.2, 6.123, -7871]})

        checks = [(input_data_df, 'int_col', 0, pd.Series([1, 2, 3, 0, 4, 0, 0])),
                  (input_data_df, 'int_col', 5, pd.Series([1, 2, 3, 5, 4, 0, 5])),
                  (input_data_df, 'int_col', -2.1, pd.Series([1.0, 2.0, 3.0, -2.1, 4.0, 0.0, -2.1])),

                  (input_data_df, 'int_no_null_col', 0, pd.Series([0, 1, 2, 3, 4, 5, 6])),
                  (input_data_df, 'int_no_null_col', 3, pd.Series([0, 1, 2, 3, 4, 5, 6])),
                  (input_data_df, 'int_no_null_col', -2, pd.Series([0, 1, 2, 3, 4, 5, 6])),

                  (input_data_df, 'str_col', 'E', pd.Series(['A', '', 'E', 'sds', 'TRsd', 'AT_Aa', ' '])),
                  (input_data_df, 'str_col', 'a', pd.Series(['A', '', 'a', 'sds', 'TRsd', 'AT_Aa', ' '])),
                  (input_data_df, 'str_col', '0', pd.Series(['A', '', '0', 'sds', 'TRsd', 'AT_Aa', ' '])),

                  (input_data_df, 'str_no_null_col', 'E', pd.Series(['1', 'R', 'y', 'Ui', '@sd', 'ty_72', 'None'])),
                  (input_data_df, 'str_no_null_col', 'a', pd.Series(['1', 'R', 'y', 'Ui', '@sd', 'ty_72', 'None'])),
                  (input_data_df, 'str_no_null_col', '0', pd.Series(['1', 'R', 'y', 'Ui', '@sd', 'ty_72', 'None'])),

                  (input_data_df, 'float_col', 0, pd.Series([3.1, -1.1, 2, 0, 0, 6.123, -7871])),
                  (input_data_df, 'float_col', 3.1, pd.Series([3.1, -1.1, 2, 0, 3.1, 6.123, -7871])),
                  (input_data_df, 'float_col', -0.2, pd.Series([3.1, -1.1, 2, 0, -0.2, 6.123, -7871])),

                  (input_data_df, 'float_no_null_col', 0, pd.Series([3.1, -1.1, 2, 0, 3.2, 6.123, -7871])),
                  (input_data_df, 'float_no_null_col', 3.1, pd.Series([3.1, -1.1, 2, 0, 3.2, 6.123, -7871])),
                  (input_data_df, 'float_no_null_col', -0.2, pd.Series([3.1, -1.1, 2, 0, 3.2, 6.123, -7871]))]
        return checks

    @staticmethod
    def data_dummify_categorical():
        input_data_df = pd.DataFrame(data={'A': ['a', 'b', 'a', 'a', 'a'],
                                           'B': ['b', 'a', 'c', 'c', np.nan],
                                           'C': [1, np.nan, 2, 3, -7.2]})
        # Metadata from expected outcome is returned as the tuple
        # (expected prefix column, unique values, row indices per values, unchanged column list)
        checks = [(input_data_df, 'A', 'dummy', '_', 1, ('dummy_', ['a', 'b'], [[0, 2, 3, 4], [1]], ['B', 'C'])),
                  (input_data_df, 'B', 'dummy', '_', 1, ('dummy_', ['a', 'b', 'c'], [[1], [0], [2, 3]], ['A', 'C'])),
                  (input_data_df, 'A', 'dummy', '', 1, ('dummy', ['a', 'b'], [[0, 2, 3, 4], [1]], ['B', 'C'])),
                  (input_data_df, 'A', '', '', 1, ('', ['a', 'b'], [[0, 2, 3, 4], [1]], ['B', 'C'])),
                  (input_data_df, 'B', '', '', 3, ('', ['a', 'b', 'c'], [[1], [0], [2, 3]], ['A', 'C'])),
                  (input_data_df, 'B', 'col', '@', -4, ('col@', ['a', 'b', 'c'], [[1], [0], [2, 3]], ['A', 'C']))]
        return checks

    @staticmethod
    def data_women_children_first_rule():
        input_data_df = pd.DataFrame(data={'gender': ['M', 'male', 'F', 'female', 'M', 'male', 'female'],
                                           'age': [5, 20, 23, 3, 41, 5, 80],
                                           'other': ['_sd', 'rt', 'R.T. 6qpoD', 'M. Smith', 'lds8', '5', 'rt'],
                                           })

        checks = [(input_data_df, 'age', 'gender', 'female', "new_col1", 1, pd.Series([1, 0, 0, 1, 0, 1, 1])),
                  (input_data_df, 'age', 'gender', 'female', "new_col2", 6, pd.Series([6, 0, 0, 6, 0, 6, 6])),
                  (input_data_df, 'age', 'gender', 'female', "new_col3", -2, pd.Series([-2, 0, 0, -2, 0, -2, -2])),
                  (input_data_df, 'age', 'gender', 'F', "new_col4", 1.1, pd.Series([1.1, 0, 1.1, 1.1, 0, 1.1, 0])),
                  (input_data_df, 'other', 'gender', 'female', "new_col5", 1, pd.Series([0, 0, 0, 0, 0, 0, 0])),
                  (input_data_df, 'age', 'other', 'female', "new_col@", 1, pd.Series([1, 0, 0, 1, 0, 1, 0])),
                  (input_data_df, 'age_2', 'gender', 'female', "new_col%", 1, pd.Series([0, 0, 0, 0, 0, 0, 0]))
                  ]
        return checks

    @staticmethod
    def data_preprocess():
        input_data_df_1 = pd.DataFrame(data={'name': ["A", "B", "C", "D", "E"],
                                             'gender': ["F", "F", "M", "F", "M"],
                                             'age': [None, 2, 28, 30, 4],
                                             'fare': [12, 34, None, 52, 1],
                                             'Pclass': [1, 2, 3, 1, 2],
                                             'Embarked': ["c", "s", "s", "s", "c"],
                                             'Survived': [1, 0, 0, 0, 1]
                                             })

        input_data_df_2 = pd.DataFrame(data={'name': ["A", "B", "C", "D", "E"],
                                             'gender': ["F", "U", "M", "U", "M"],
                                             'age': [None, 2, None, 30, 4],
                                             'fare': [12, 34, 23.8, 52, 1],
                                             'Pclass': [1, 2, 3, 1, 2],
                                             'Embarked': ["c", "s", "s", "s", "c"],
                                             'Survived': [1, 0, 0, 0, 1]
                                             })

        input_data_df_3 = pd.DataFrame(data={'name': ["A", "B", "C", "D", "E"],
                                             'gender': ["female", "female", "male", "male", "male"],
                                             'age': [None, 22, None, 30, 4],
                                             'fare': [12, 34, 23.8, 52, 1],
                                             'Pclass': [1, 2, 3, 1, 2],
                                             'Embarked': ["c", "s", "s", "s", "c"],
                                             'Survived': [1, 0, 0, 0, 1]
                                             })

        checks = [(input_data_df_1, 'age', 'gender', ['fare', 'age', 'Pclass'], 30, 'F', 'women_child', 2, 3,
                  pd.DataFrame(data={'fare': [12, 34, None, 52, 1],
                                     'age': [30, 2, 28, 30, 4],
                                     'Pclass': [1, 2, 3, 1, 2],
                                     'women_child': [2, 2, 0, 2, 2],
                                     'F': [3, 3, 0, 3, 0]
                                     })),
                  (input_data_df_2, 'age', 'gender', ['age', 'Pclass'], 21, 'F', 'women_child', 2, 3,
                   pd.DataFrame(data={'age': [21, 2, 21, 30, 4],
                                      'Pclass': [1, 2, 3, 1, 2],
                                      'women_child': [2, 2, 0, 0, 2],
                                      'F': [3, 0, 0, 0, 0],
                                      'M': [0, 0, 3, 0, 3]})),
                  (input_data_df_3, 'age', 'gender', ['age', 'Pclass'], 3.5, 'female', 'women_child', 2, 13,
                   pd.DataFrame(data={'age': [3.5, 22, 3.5, 30, 4],
                                      'Pclass': [1, 2, 3, 1, 2],
                                      'women_child': [2, 2, 2, 0, 2],
                                      'female': [13, 13, 0, 0, 0]}))
                  ]
        # input_df, age_col, gender_col, fixed_columns, fill_na_default_value, female_gender_value,
        #     children_women_first_rule_column_name, children_women_first_rule_scale, dummy_scale, expected_result
        return checks

    @staticmethod
    def clean_dataframe():
        input_data_df = pd.DataFrame(data={'A': [1, 2, 3, 6, 4, 0, None],
                                           'B': [0, 1, 2, 3, 4, 5, 6],
                                           'C': ['A', '', None, 'sds', 'TRsd', 'AT_Aa', ' '],
                                           'D': ['1', 'R', 'y', 'Ui', '@sd', 'ty_72', 'None'],
                                           'E': [3.1, -1.1, 2, 0, None, 6.123, -7871],
                                           'F': [3.1, -1.1, 2, 0, 3.2, 6.123, -7871]})

        # children_women_first_rule_column_name,  fixed_columns: list, gender_value_list, expected

        # fixed_columns + [children_women_first_rule_column_name] + gender_value_list
        checks = [(input_data_df, 'D', ['A', 'B', 'C'], ['E', 'F'], deepcopy(input_data_df)),
                  (input_data_df, 'A', [], ['D'], deepcopy(input_data_df[['A', 'D']])),
                  (input_data_df, 'C', ['D'], [], deepcopy(input_data_df[['D', 'C']])),
                  (input_data_df, 'D', [], [],  deepcopy(input_data_df[['D']])),
                  (input_data_df, 'D', ['E'], ['F'], deepcopy(input_data_df[['E', 'D', 'F']]))
                  ]

        return checks


titanic_preprocessing = TitanicPreprocessingDataTest()


def preprocessing_data_fill_na():
    return titanic_preprocessing.data_fill_na()


# @pytest.fixture(scope='function')
def preprocessing_data_dummify_categorical():
    return titanic_preprocessing.data_dummify_categorical()


def preprocessing_data_women_children_first_rule():
    return titanic_preprocessing.data_women_children_first_rule()


def preprocessing_data_preprocess():
    return titanic_preprocessing.data_preprocess()


def preprocessing_clean_dataframe():
    return titanic_preprocessing.clean_dataframe()
