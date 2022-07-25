import reports

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def delete_duplicate_rows_by_attribute(data_frame: pd.DataFrame, attribute_name: str, print_report: bool = False) -> pd.DataFrame:
    """Delete duplicate rows from the DataFrame, given the attribute name, passed as parameter.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        attribute_name (str): Attribute name to be used to delete duplicates.
        print_report (bool, optional): Print the report. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame with deleted duplicate rows.
    """

    print('\n*****INICIO DELETE DUPLICATE ROWS BY ATTRIBUTE******')
    rows_duplicated = data_frame[data_frame.duplicated(
        attribute_name, keep=False)].sort_values(attribute_name)
    if rows_duplicated.size > 0:
        print('Linhas duplicadas encontradas para o atributo {}.'.format(
            attribute_name))
        if print_report:
            reports.duplicate_rows_by_attribute(
                data_frame=data_frame, rows_duplicated=rows_duplicated, attribute=attribute_name)
        data_frame.drop_duplicates(
            subset=attribute_name, keep='first', inplace=True)
    print('*****FIM DELETE DUPLICATE ROWS BY ATTRIBUTE*********')
    return data_frame


def delete_nan_rows(data_frame: pd.DataFrame, print_report: bool = False) -> pd.DataFrame:
    """Delete nan rows from the DataFrame.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        print_report (bool, optional): Print the report. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame with deleted nan rows.
    """
    print('\n*****INICIO DELETE NAN ROWS******')
    nan_rows = data_frame.isna().sum().sum()
    if nan_rows > 0:
        print('Linhas com valores NaN encontradas.')
        if print_report:
            reports.nan_attributes(data_frame=data_frame, total_nan=nan_rows)
        data_frame.dropna(inplace=True)
    else:
        print('Nenhuma linha com valores NaNN encontrada.')
    print('*****FIM DELETE NAN ROWS*********')
    return data_frame


def label_encoder_columns(data_frame: pd.DataFrame, columns_label_encoded: dict, columns_names: list) -> tuple:
    """Label encode the DataFrame, given the columns names, passed as parameters.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        columns_label_encoded (dict): Dictionary with the label encoders for each column.
        columns_names (list): Array of strings with columns names to label encode.

    Returns:
        tuple: A tuple with the encoded data_frame and the columns_label_encoded, in this order.
    """
    print('\n*****INICIO LABEL ENCODER******')
    for column in columns_names:
        if column in data_frame.columns:
            if column not in columns_label_encoded:
                encoder_column = LabelEncoder()
                data_frame[column] = encoder_column.fit_transform(
                    data_frame[column])
                columns_label_encoded[column] = encoder_column
            else:
                print('!!!>>> A coluna {} já está codificada.'.format(column))
        else:
            print(
                '!!!>>> Coluna {} não encontrada no DataFrame para label encoding.'.format(column))
    print('*****FIM LABEL ENCODER*********')
    return data_frame, columns_label_encoded


def drop_feature_by_correlation(data_frame: pd.DataFrame, method: str, columns_names: list, threshold: float = 0.9) -> pd.DataFrame:
    """Drop the features by correlation, given the columns names, passed as parameters.
    The threshold parameter is used to define the threshold of correlation.
    The default value is 0.9.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        method (str): Method to be used to calculate the correlation.
        columns_names (list): Array of strings with columns names to drop.
        threshold (float, optional): Threshold of correlation. Defaults to 0.9.

    Returns:
        pd.DataFrame: A DataFrame with dropped features.
    """
    # TODO: Verify de function in execution, if not OK look the Albon book (10.3 Handling Highly Correlated Features).
    print('\n*****INICIO DROP FEATURE BY CORRELATION******')
    for column in columns_names:
        if column in data_frame.columns:
            correlation_matrix = data_frame.corr(method=method)
            correlation_matrix = correlation_matrix.loc[column, :]
            correlation_matrix = correlation_matrix.sort_values(
                ascending=False)
            correlation_matrix = correlation_matrix[correlation_matrix > threshold]
            correlation_matrix = correlation_matrix.drop(column)
            correlation_matrix = correlation_matrix.dropna()
            correlation_matrix = correlation_matrix.index.tolist()
            if len(correlation_matrix) > 0:
                print('>>> Correlação entre a coluna {} e as colunas {} foi encontrada.'.format(
                    column, correlation_matrix))
                data_frame.drop(columns=correlation_matrix, inplace=True)
            else:
                print(
                    '>>> Nenhuma correlação entre a coluna {} e as colunas encontrada.'.format(column))
        else:
            print(
                '!!!>>> Coluna {} não encontrada no DataFrame para drop feature by correlation.'.format(column))
    print('*****FIM DROP FEATURE BY CORRELATION*********')
    return data_frame
