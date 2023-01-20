import pytest
import conftest
from pandas import DataFrame, Series
from typing import Union
from src.preprocessing import titanic_preprocessing as tp


@pytest.mark.parametrize('input_df, col, prefix, prefix_sep, scale, expected_result_metadata',
                         conftest.TitanicPreprocessingDataTest.data_dummify_categorical)
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

    assert isinstance(output_df, DataFrame)
    assert id(output_df) != id(input_df)
    # TODO correct wrt to expected_result_metadata
    """
    dummy_columns = [c for c in output_df if c.startswith(f'{prefix}{prefix_sep}')]
    assert len(dummy_columns) == input_df[col].nunique()
    assert col not in output_df.columns
    for c in output_df:
        if c in dummy_columns:
            assert 0 in set(output_df[c])
            assert scale in set(output_df[c])
        else:
            assert (sum([input_df[col][i] is None or output_df[col][i] == pytest.approx(input_df[col][i])
                         for i in range(len(input_df))]
                        ) == len(input_df)
                    )
    """


@pytest.mark.parametrize('input_df, col, default_value, expected_result',
                         conftest.TitanicPreprocessingDataTest.data_fill_na)
def test_fill_na(input_df: DataFrame,
                 col: str,
                 default_value: Union[str, float, int],
                 expected_result: Series):
    output_df = tp.fill_na(df=input_df, col=col, default_value=default_value)

    # output_df is expected to be a dataframe
    assert isinstance(output_df, DataFrame)
    # ids object shall not be the same
    assert(id(input_df) != id(output_df))
    # the columns' name in output shall match those from input
    assert(sum([int(c in input_df.columns) for c in output_df.columns]) == len(input_df))
    # the dataframe length shall remain unchanged
    assert(len(output_df) == len(input_df))

    for c in input_df.columns:
        if c == col:
            assert(output_df[col] == pytest.approx(expected_result))
        else:
            # other columns shall remain unchanged
            assert c in input_df.columns
            assert(sum([input_df[col][i] is None or output_df[col][i] == pytest.approx(input_df[col][i])
                       for i in range(len(input_df))]
                       ) == len(input_df)
                   )


@pytest.mark.parametrize('input_df, age_col, gender_col, female_value, new_col_name, scale, expected_result',
                         conftest.TitanicPreprocessingDataTest.data_women_children_first_rule)
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
    # ids object shall not be the same
    assert id(output_df) != id(input_df)
    # new_col_name must be a column from output_df
    assert new_col_name in output_df.columns

    for c in output_df.columns:
        if c == new_col_name:
            assert(output_df[new_col_name] == expected_result)
        else:
            # other columns shall remain unchanged
            assert c in input_df.columns
            assert(sum([output_df[c][i] == pytest.approx(input_df[c][i])
                        for i in range(len(input_df))]) == len(input_df)
                   )


@pytest.skip
@pytest.mark.parametrize(
    """input_df, age_col, gender_col, fixed_columns, fill_na_default_value, female_gender_value, 
    children_women_first_rule_column_name, children_women_first_rule_scale, dummy_scale, expected_result""",
    conftest.TitanicPreprocessingDataTest.data_preprocess)
def test_preprocess(input_df: DataFrame,
                    age_col: str,
                    gender_col: str,
                    fixed_columns: list,
                    fill_na_default_value: float,
                    female_gender_value: str,
                    children_women_first_rule_column_name: str,
                    children_women_first_rule_scale,
                    dummy_scale: int,
                    expected_result: DataFrame
                    ):
    pass
