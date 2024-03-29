from pandas import DataFrame, get_dummies
from typing import Union
from copy import deepcopy


def dummify_categorical(df: DataFrame,
                        col: str,
                        prefix: str = '',
                        prefix_sep: str = '',
                        scale: int = 1
                        ) -> DataFrame:
    values = df[col].unique()
    df_dummify = get_dummies(df, prefix=prefix, prefix_sep=prefix_sep, columns=[col])
    new_cols = [f'{prefix}{prefix_sep}{val}' for val in values if val == val]
    df_dummify[new_cols] = df_dummify[new_cols].mul(scale)
    return df_dummify


def fill_na(df: DataFrame,
            col: str,
            default_value: Union[str, int, float]
            ) -> DataFrame:
    dg = deepcopy(df)
    dg[col].fillna(value=default_value, inplace=True)
    return dg


def women_children_first_rule(df: DataFrame,
                              age_col: str,
                              gender_col: str,
                              female_value: Union[str, int],
                              new_col_name: str,
                              scale: int = 1
                              ) -> DataFrame:
    dg = deepcopy(df)
    if age_col not in dg.columns or gender_col not in dg.columns:
        dg[new_col_name] = [0]*len(dg)
    elif dg[age_col].dtypes not in (int, float):
        dg[new_col_name] = [0]*len(dg)
    else:
        dg[new_col_name] = dg[[age_col, gender_col]].apply(
            lambda r: scale*int(r[0] < 18 or r[1] == female_value) if type(r[0]) is int or type(r[0]) is float else 0,
            axis=1)
    return dg


def clean_dataframe(df: DataFrame,
                    children_women_first_rule_column_name: str,
                    fixed_columns: list,
                    gender_value_list: list) -> DataFrame:
    dg = deepcopy(df)
    return dg[fixed_columns + [children_women_first_rule_column_name] + gender_value_list]


def preprocess(df: DataFrame,
               age_col: str,
               gender_col: str,
               fixed_columns: list,
               fill_na_default_value: float = 29.6,
               female_gender_value: str = 'female',
               children_women_first_rule_column_name: str = 'women_children_first_rule_eligible',
               children_women_first_rule_scale: int = 5,
               dummy_scale: int = 4
               ) -> DataFrame:

    gender_values = list(df[gender_col].sort_values(ascending=True).unique())
    # a small analysis showed that the avg age is around 29.6
    dg = deepcopy(df)
    dg = fill_na(df=dg, default_value=fill_na_default_value, col=age_col)
    dg = women_children_first_rule(df=dg,
                                   age_col=age_col,
                                   gender_col=gender_col,
                                   female_value=female_gender_value,
                                   new_col_name=children_women_first_rule_column_name,
                                   scale=children_women_first_rule_scale)
    dg = dummify_categorical(df=dg, col=gender_col, scale=dummy_scale)

    clean_df = clean_dataframe(df=dg,
                               fixed_columns=fixed_columns,
                               children_women_first_rule_column_name=children_women_first_rule_column_name,
                               gender_value_list=gender_values[:-1]
                               )

    return clean_df
