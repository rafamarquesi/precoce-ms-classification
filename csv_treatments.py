import pandas as pd
import reports

pd.set_option('float_format', '{:f}'.format)


def delete_columns(data_frame: pd.DataFrame, delete_columns_names: list) -> pd.DataFrame:
    """Delete columns from the DataFrame, given the columns names, passed as parameters.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        delete_columns_names (list): Array of strings with columns names to exclude.

    Returns:
        pd.DataFrame: A DataFrame with deleted columns.
    """

    print('*****INICIO DELETE COLUNAS******')
    for column in delete_columns_names:
        if column in data_frame.columns:
            data_frame.drop(column, axis='columns', inplace=True)
            print('Coluna {} excluída.'.format(column))
        else:
            print(
                '!!!>>> Coluna " {} " não encontrada no DataFrame para exclusão.'.format(column))
    print('*****FIM DELETE COLUNAS*********')
    return data_frame


def load_data(csv_path: str, sep: str = ';', encoding: str = 'latin1', decimal: str = ',', delete_columns_names: list = None, number_csv_lines: int = None, dtype_dict: dict = None, parse_dates: list = None) -> pd.DataFrame:
    """Load the CSV file, given the path, column names, and number of rows to load, passed as parameters.

    Args:
        csv_path (str): Path where the CSV file is located.
        str (str, optional): Separator of the CSV file. Defaults to ';'.
        encoding (str, optional): Encoding of the CSV file. Defaults to 'latin1'.
        decimal (str, optional): Decimal separator of the CSV file. Defaults to ','.
        delete_columns_names (list, optional): Array of strings with columns names to exclude. Defaults to None.
        number_csv_lines (int, optional): Number of lines that will be loaded from the CSV file. Defaults to None.
        dtype_dict (dict, optional): Dictionary with the types of the columns. Defaults to None.
        parse_dates (list, optional): Array of strings with columns names to parse as dates. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame with the data loaded from the CSV file.
    """

    print('\n*****INICIO LOAD DATA******')
    if number_csv_lines != None:
        temp_data = pd.read_csv(
            csv_path, sep=sep, encoding=encoding, decimal=decimal,
            nrows=number_csv_lines, dtype=dtype_dict, parse_dates=parse_dates
        )
    else:
        temp_data = pd.read_csv(
            csv_path, sep=sep, encoding=encoding, decimal=decimal, dtype=dtype_dict, parse_dates=parse_dates
        )

    reports.informations(temp_data)
    # Used to print min and max of each column
    # reports.print_min_max_column(temp_data)

    if delete_columns_names != None:
        temp_data = delete_columns(
            data_frame=temp_data, delete_columns_names=delete_columns_names
        )

    reports.informations(temp_data)
    print('*****FIM LOAD DATA******')

    return temp_data


def generate_new_csv(data_frame: pd.DataFrame, csv_path: str, sep: str = ';', encoding: str = 'latin1', index: bool = False) -> None:
    """Generate a new CSV file, given the DataFrame and the path, passed as parameters.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        csv_path (str): Path where the CSV file will be generated.
        sep (str, optional): Separator of the CSV file. Defaults to ';'.
        encoding (str, optional): Encoding of the CSV file. Defaults to 'latin1'.
        index (bool, optional): If True, the index will be included in the CSV file. Defaults to False.
    """
    data_frame.to_csv(csv_path, sep=sep, encoding=encoding, index=index)
    print('\nCSV gerado com sucesso em: {}.'.format(csv_path))


def move_cloumns_last_positions(data_frame: pd.DataFrame, columns_names: list) -> pd.DataFrame:
    """Move columns to the last positions of the DataFrame, given the position of the column names in the array, passed as parameter.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        columns_names (list): Columns names to be moved.

    Returns:
        pd.DataFrame: A DataFrame with the columns moved to the last positions.
    """
    data_frame = data_frame[[c for c in data_frame if c not in columns_names] + [
        c for c in columns_names if c in data_frame]]
    # data_frame = data_frame.reindex(
    #     columns=data_frame.columns.tolist() + columns_names)
    return data_frame
