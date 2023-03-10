import numpy as np
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'src')))

import pytest
# import tests.conftest as conftest
import tests.parameters_titanic_preprocessing as param_preprocess
from pandas import DataFrame, Series, testing
from typing import Union
from preprocessing import titanic_preprocessing as tp


@pytest.mark.parametrize('input_df, col, prefix, prefix_sep, scale, expected_result_metadata',
                         param_preprocess.preprocessing_data_dummify_categorical())
def test_dummify_categorical(input_df: DataFrame,
                             col: str,
                             prefix: str,
                             prefix_sep: str,
                             scale: int,
                             expected_result_metadata: tuple):
    output_df = tp.dummify_categorical(df=input_df,
                                       col=col,
                                       prefix=prefix,
                                       prefix_sep=prefix_sep,
                                       scale=scale)

    assert isinstance(output_df, DataFrame), "output is not a Dataframe"
    assert id(output_df) != id(input_df), 'object ids are the same'
    assert output_df.shape[0] == input_df.shape[0], 'output_df and input_df do not have the same length'

    # Metadata from expected outcome is returned as the tuple
    # (expected prefix column, unique values, row indices per values, unchanged column list)
    expected_prefix, unique_values, row_indices, unchanged_columns = expected_result_metadata
    dummy_columns = [c for c in output_df if c.startswith(expected_prefix) and c not in unchanged_columns]
    assert len(dummy_columns) == len(unique_values), 'the nb of dummy columns does not match the expected size'
    expected_dummy_colums = [f'{expected_prefix}{v}' for v in unique_values]
    col_intersection = set(expected_dummy_colums).intersection(set(dummy_columns))
    assert len(col_intersection) == len(unique_values), 'dummy columns do not match with expected columns'
    assert col not in output_df.columns
    for c in unchanged_columns:
        assert c in output_df.columns, f'{c} not in output column'

    for v, indices in zip(unique_values, row_indices):
        dummy_col = f'{expected_prefix}{v}'
        expected_col = np.zeros(output_df.shape[0])
        expected_col[indices] = scale
        assert output_df[dummy_col].to_numpy() == pytest.approx(expected_col), f'value mismatch'


@pytest.mark.parametrize('input_df, col, default_value, expected_result', param_preprocess.preprocessing_data_fill_na())
def test_fill_na(input_df: DataFrame,
                 col: str,
                 default_value: Union[str, float, int],
                 expected_result: Series,
                 ):
    output_df = tp.fill_na(df=input_df, col=col, default_value=default_value)

    # output_df is expected to be a dataframe
    assert isinstance(output_df, DataFrame)
    assert output_df.shape[0] == input_df.shape[0]
    # ids object shall not be the same
    assert(id(input_df) != id(output_df))
    # col must be a column from output_df
    assert col in output_df.columns
    # the columns' name in output shall match those from input
    assert(sum([int(c in input_df.columns) for c in output_df.columns]) == len(input_df.columns))
    # the dataframe length shall remain unchanged
    assert(len(output_df) == len(input_df))

    for c in output_df.columns:
        if c == col:
            # assert output_df[col].to_numpy() == pytest.approx(expected_result.to_numpy()) # works only with numbers
            testing.assert_series_equal(left=output_df[col], right=expected_result, check_dtype=False,
                                        check_names=False)
        else:
            # other columns shall remain unchanged
            assert c in input_df.columns
            assert(sum([input_df[c][i] != input_df[c][i] or output_df[c][i] == pytest.approx(input_df[c][i])
                       for i in range(len(input_df))]
                       ) == len(input_df)
                   )


@pytest.mark.parametrize('input_df, age_col, gender_col, female_value, new_col_name, scale, expected_result',
                         param_preprocess.preprocessing_data_women_children_first_rule())
def test_women_children_first_rule(input_df: DataFrame,
                                   age_col: str,
                                   gender_col: str,
                                   female_value: Union[str, int],
                                   new_col_name: str,
                                   scale: int,
                                   expected_result: Series
                                   ):
    output_df = tp.women_children_first_rule(df=input_df,
                                             age_col=age_col,
                                             gender_col=gender_col,
                                             female_value=female_value,
                                             new_col_name=new_col_name,
                                             scale=scale)
    # output_df is expected to be a dataframe
    assert isinstance(output_df, DataFrame)
    assert output_df.shape[0] == input_df.shape[0]
    # ids object shall not be the same
    assert id(output_df) != id(input_df)
    # new_col_name must be a column from output_df
    assert new_col_name in output_df.columns

    for c in output_df.columns:
        if c == new_col_name:
            # output is always a binary Series => pytest.approx can be used
            assert output_df[new_col_name].to_numpy() == pytest.approx(expected_result.to_numpy())
            # testing.assert_series_equal(left=output_df[new_col_name], right=expected_result, check_dtype=False,
            #                            check_names=False)
        else:
            # other columns shall remain unchanged
            assert c in input_df.columns
            assert(sum([output_df[c][i] == pytest.approx(input_df[c][i])
                        for i in range(len(input_df))]) == len(input_df)
                   )


# TODO : write test
@pytest.skip("test not written yet")
@pytest.mark.parametrize(
    """input_df, children_women_first_rule_column_name, fixed_columns, gender_value_list, expected_result""",
    param_preprocess.preprocessing_clean_dataframe())
def test_clean_dataframe(input_df: DataFrame,
                         children_women_first_rule_column_name: str,
                         fixed_columns: list,
                         gender_value_list: list,
                         expected_result: DataFrame):
    clean_df = tp.clean_dataframe(df=input_df,
                                  children_women_first_rule_column_name=children_women_first_rule_column_name,
                                  fixed_columns=fixed_columns,
                                  gender_value_list=gender_value_list
                                  )

    ref_set = set(fixed_columns).union({children_women_first_rule_column_name}).union(set(gender_value_list))
    assert id(clean_df) != id(input_df)

    assert len(set(clean_df.columns).intersection(ref_set)) == len(clean_df.columns)
    assert len(set(clean_df.columns).intersection(ref_set)) == len(ref_set)

    testing.assert_frame_equal(left=clean_df, right=expected_result, check_dtype=False, check_names=False)


@pytest.mark.parametrize(
    """input_df, age_col, gender_col, fixed_columns, fill_na_default_value, female_gender_value,
    children_women_first_rule_column_name, children_women_first_rule_scale, dummy_scale, expected_result""",
    param_preprocess.preprocessing_data_preprocess())
def test_preprocess(input_df: DataFrame,
                    age_col: str,
                    gender_col: str,
                    fixed_columns: list,
                    fill_na_default_value: float,
                    female_gender_value: str,
                    children_women_first_rule_column_name: str,
                    children_women_first_rule_scale: int,
                    dummy_scale: int,
                    expected_result: DataFrame
                    ):
    output_df = tp.preprocess(df=input_df,
                              age_col=age_col,
                              gender_col=gender_col,
                              fixed_columns=fixed_columns,
                              fill_na_default_value=fill_na_default_value,
                              female_gender_value=female_gender_value,
                              children_women_first_rule_column_name=children_women_first_rule_column_name,
                              children_women_first_rule_scale=children_women_first_rule_scale,
                              dummy_scale=dummy_scale)
    # id checks
    assert id(output_df) != id(input_df)
    # size checks
    assert output_df.size == expected_result.size
    # columns checks
    assert len(set(output_df.columns).intersection(set(expected_result.columns))) == len(expected_result.columns)
    # content checks
    # for col in output_df.columns:
    #    testing.assert_series_equal(left=output_df[col], right=expected_result[col], check_dtype=False,
    #                                check_names=False)

    # directly on dataframe
    testing.assert_frame_equal(left=output_df, right=expected_result, check_dtype=False, check_names=False)
