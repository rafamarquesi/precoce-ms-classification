import reports

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, OrdinalEncoder


def delete_columns(data_frame: pd.DataFrame, columns_names: list) -> pd.DataFrame:
    """Delete columns from the DataFrame, given the columns names, passed as parameters.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        columns_names (list): Array of strings with columns names to delete.

    Returns:
        pd.DataFrame: A DataFrame with deleted columns.
    """
    print('\n*****INICIO DELETE COLUMNS******')
    for column in columns_names:
        if column in data_frame.columns:
            data_frame.drop(column, axis=1, inplace=True)
            print('Coluna {} deletada.'.format(column))
        else:
            print(
                '!!!>>> Coluna {} não encontrada no DataFrame para delete.'.format(column))
    print('*****FIM DELETE COLUMNS*********')
    return data_frame


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
    if attribute_name in data_frame.columns:
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
    else:
        print(
            '!!!>>> Atributo {} não encontrado no DataFrame para delete duplicates rows by attibute.'.format(attribute_name))
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


def ordinal_encoder_columns(data_frame: pd.DataFrame, columns_ordinal_encoded: dict, columns_names: dict) -> tuple:
    """Ordinal encode the DataFrame, given the columns names, passed as parameters.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        columns_ordinal_encoded (dict): Dictionary with the ordinal encoders for each column.
        columns_names (dict): Dictionary with column name, as key, and order of the categories. To keep the original order of the categories pass None as value.

    Returns:
        tuple: A tuple with the encoded data_frame and the columns_ordinal_encoded, in this order.
    """
    print('\n*****INICIO ORDINAL ENCODER******')
    for column, categories in columns_names.items():
        if column in data_frame.columns:
            if column not in columns_ordinal_encoded:
                if categories is None:
                    encoder_column = OrdinalEncoder()
                else:
                    encoder_column = OrdinalEncoder(categories=[categories])
                data_frame[column] = encoder_column.fit_transform(
                    data_frame[[column]])
                columns_ordinal_encoded[column] = encoder_column
            else:
                print('!!!>>> A coluna {} já está codificada.'.format(column))
        else:
            print(
                '!!!>>> Coluna {} não encontrada no DataFrame para ordinal encoding.'.format(column))
    print('*****FIM ORDINAL ENCODER*********')
    return data_frame, columns_ordinal_encoded


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


def one_hot_encoder_columns(data_frame: pd.DataFrame, columns_one_hot_encoded: dict, columns_names: list) -> tuple:
    """One-hot encode the DataFrame, given the columns names, passed as parameters.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        columns_one_hot_encoded (dict): Dictionary with the one-hot encoders for each column.
        columns_names (list): Array of strings with columns names to one-hot encode.

    Returns:
        tuple: A tuple with the encoded data_frame and the columns_one_hot_encoded, in this order.
    """
    print('\n*****INICIO ONE-HOT ENCODER******')
    for column in columns_names:
        if column in data_frame.columns:
            if column not in columns_one_hot_encoded:
                encoder_column = OneHotEncoder(
                    sparse=False, handle_unknown='infrequent_if_exist', drop='if_binary')
                encoded_df = pd.DataFrame(encoder_column.fit_transform(
                    data_frame[[column]]), columns=encoder_column.get_feature_names_out(), index=data_frame.index.values.tolist())
                data_frame = pd.concat([data_frame, encoded_df], axis=1)
                data_frame = delete_columns(
                    data_frame=data_frame, columns_names=[column])
                columns_one_hot_encoded[column] = encoder_column
            else:
                print('!!!>>> A coluna {} já está codificada.'.format(column))
        else:
            print(
                '!!!>>> Coluna {} não encontrada no DataFrame para one-hot encoding.'.format(column))
    print('*****FIM ONE-HOT ENCODER*********')
    return data_frame, columns_one_hot_encoded


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


def min_max_scaler_columns(data_frame: pd.DataFrame, columns_min_max_scaled: dict, columns_names: list) -> tuple:
    """Min-Max scale the DataFrame, given the columns names, passed as parameters.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        columns_min_max_scaled (dict): Dictionary with the min-max scalers for each column.
        columns_names (list): Array of strings with columns names to scale.

    Returns:
        tuple: A tuple with the scaled data_frame and the columns_scalers, in this order.
    """
    print('\n*****INICIO MIN-MAX SCALER******')
    for column in columns_names:
        if column in data_frame.columns:
            if column not in columns_min_max_scaled:
                scaler_column = MinMaxScaler()
                # data_frame[column] = scaler_column.fit_transform(
                #     data_frame[column])
                data_frame[column] = scaler_column.fit_transform(
                    data_frame[column].values.reshape(-1, 1))
                columns_min_max_scaled[column] = scaler_column
            else:
                print('!!!>>> A coluna {} já está normalizada.'.format(column))
        else:
            print(
                '!!!>>> Coluna {} não encontrada no DataFrame para min-max scaler.'.format(column))
    print('*****FIM MIN-MAX SCALER*********')
    return data_frame, columns_min_max_scaled


def detect_outliers(series: pd.Series, whis: float = 1.5) -> pd.Series:
    """Detect outliers in a series.

    Args:
        series (pd.Series): Series to be treated.
        whis (float, optional): Whisker value. Defaults to 1.5.

    Returns:
        pd.Series: A Series with the outliers detected.
    """

    q75, q25 = np.percentile(series, [75, 25])
    iqr = q75 - q25
    return ~((series - series.median()).abs() <= (whis * iqr))
