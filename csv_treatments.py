import pandas as pd
import reports
import utils

pd.set_option('float_format', '{:f}'.format)


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
    # reports.min_max_column(temp_data)

    if delete_columns_names != None:
        temp_data = utils.delete_columns(
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
