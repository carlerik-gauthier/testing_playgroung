import pytest
import conftest
from pandas import DataFrame, Series
from typing import Union
from src.preprocessing import titanic_preprocessing as tp


def test_dummify_categorical():
    pass


@pytest.mark.parametrize('input_df, col, default_value, expected_result',
                         conftest.TitanicPreprocessingDataTest.data_fill_na)
def test_fill_na(input_df: DataFrame,
                 col: str,
                 default_value: Union[str, float, int],
                 expected_result: Series):
    output_df = tp.fill_na(df=input_df, col=col, default_value=default_value)
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
            assert(sum([input_df[col][i] is None or output_df[col][i] == pytest.approx(input_df[col][i])
                       for i in range(len(input_df))]
                       ) == len(input_df)
                   )


def test_women_children_first_rule():
    pass


def test_preprocess():
    pass
