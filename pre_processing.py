import pandas as pd


def delete_duplicate_rows_by_attribute(data_frame: pd.DataFrame, attribute_name: str) -> pd.DataFrame:
    """Delete duplicate rows from the DataFrame, given the attribute name, passed as parameter.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        attribute_name (str): Attribute name to be used to delete duplicates.

    Returns:
        pd.DataFrame: A DataFrame with deleted duplicate rows.
    """

    print('\n*****INICIO DELETE DUPLICATE ROWS BY ATTRIBUTE******')
    rows_duplicated = data_frame[data_frame.duplicated(
        attribute_name, keep=False)].sort_values(attribute_name)
    if rows_duplicated.size > 0:
        print('Linhas duplicadas encontradas para o atributo {}. Linhas duplicadas:\n'.format(
            attribute_name))
        print(rows_duplicated)
        # self.relatorio_atributo_duplicado(atributo=atributo, dados_duplicados=dados_duplicados, dados=dados_temp)
        data_frame.drop_duplicates(
            subset=attribute_name, keep='first', inplace=True)
    print('*****FIM DELETE DUPLICATE ROWS BY ATTRIBUTE*********')
    return data_frame


def delete_nan_rows(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Delete nan rows from the DataFrame.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.

    Returns:
        pd.DataFrame: A DataFrame with deleted nan rows.
    """
    print('\n*****INICIO DELETE NAN ROWS******')
    nan_rows = data_frame.isna().sum().sum()
    if nan_rows > 0:
        print('Linhas com valores nulos encontradas. Linhas com valores nulos:\n')
        print(data_frame[data_frame.isna().sum(axis=1) > 0])
        # self.relatorio_atributo_nan(total_nan=total_nan, dados=dados_temp)
        data_frame.dropna(inplace=True)
    print('*****FIM DELETE NAN ROWS*********')
    return data_frame
