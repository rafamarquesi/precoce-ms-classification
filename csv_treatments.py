import reports
import utils

import pandas as pd

pd.set_option('float_format', '{:f}'.format)


@utils.timeit
def load_data(csv_path: str, sep: str = ';', encoding: str = 'latin1', decimal: str = ',', delete_columns_names: list = None, number_csv_lines: int = None, dtype_dict: dict = None, parse_dates: list = False) -> pd.DataFrame:
    """Load the CSV file, given the path, column names, and number of rows to load, passed as parameters.

    Args:
        csv_path (str): Path where the CSV file is located.
        str (str, optional): Separator of the CSV file. Defaults to ';'.
        encoding (str, optional): Encoding of the CSV file. Defaults to 'latin1'.
        decimal (str, optional): Decimal separator of the CSV file. Defaults to ','.
        delete_columns_names (list, optional): Array of strings with columns names to exclude. Defaults to None.
        number_csv_lines (int, optional): Number of lines that will be loaded from the CSV file. Defaults to None.
        dtype_dict (dict, optional): Dictionary with the types of the columns. Defaults to None.
        parse_dates (list, optional): Array of strings with columns names to parse as dates. Defaults to False.

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


@utils.timeit
def generate_new_csv(data_frame: pd.DataFrame, csv_path: str, sep: str = ';', encoding: str = 'latin1', index: bool = False, date_format: str = '%d%b%Y', decimal: str = ',') -> None:
    """Generate a new CSV file, given the DataFrame and the path, passed as parameters.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        csv_path (str): Path where the CSV file will be generated.
        sep (str, optional): Separator of the CSV file. Defaults to ';'.
        encoding (str, optional): Encoding of the CSV file. Defaults to 'latin1'.
        index (bool, optional): If True, the index will be included in the CSV file. Defaults to False.
        date_format (str, optional): Date format of the CSV file. Defaults to '%d%b%Y'.
        decimal (str, optional): Decimal separator of the CSV file. Defaults to ','.
    """
    data_frame.to_csv(
        csv_path, sep=sep, encoding=encoding,
        index=index, date_format=date_format,
        decimal=decimal
    )
    print('\nCSV gerado com sucesso em: {}.'.format(csv_path))


def choose_csv_path(sampling: str = None, folder_path: str = None) -> str:
    """Choose the path of the CSV file to be loaded.

    Args:
        sampling (str, optional): Key name of the CSV file to be loaded. The sampling keys availables to load are '0.2', '0.5', '2', '5', '10', '20', '30', '40', '50', '60', '100', and 'after_correlation-02-11-2022_18:29:05'. Defaults to None.
        folder_path (str, optional): Path of the folder where the CSV file is located. Defaults to None.

    Returns:
        str: Path of the CSV file to be loaded.
    """

    csv_files = {
        '0.2': '{}TAB_MODELAGEM_RAFAEL_2020_1-0.2-percentage-sampling.csv'.format(folder_path),
        '0.5': '{}TAB_MODELAGEM_RAFAEL_2020_1-0.5-percentage-sampling.csv'.format(folder_path),
        '2': '{}TAB_MODELAGEM_RAFAEL_2020_1-2.0-percentage-sampling.csv'.format(folder_path),
        '5': '{}TAB_MODELAGEM_RAFAEL_2020_1-5.0-percentage-sampling.csv'.format(folder_path),
        '10': '{}TAB_MODELAGEM_RAFAEL_2020_1-10.0-percentage-sampling.csv'.format(folder_path),
        '20': '{}TAB_MODELAGEM_RAFAEL_2020_1-20.0-percentage-sampling.csv'.format(folder_path),
        '30': '{}TAB_MODELAGEM_RAFAEL_2020_1-30.0-percentage-sampling.csv'.format(folder_path),
        '40': '{}TAB_MODELAGEM_RAFAEL_2020_1-40.0-percentage-sampling.csv'.format(folder_path),
        '50': '{}TAB_MODELAGEM_RAFAEL_2020_1-50.0-percentage-sampling.csv'.format(folder_path),
        '60': '{}TAB_MODELAGEM_RAFAEL_2020_1-60.0-percentage-sampling.csv'.format(folder_path),
        '100': '{}TAB_MODELAGEM_RAFAEL_2020_1.csv'.format(folder_path),
        'after_correlation-02-11-2022_18:29:05': '{}TAB_MODELAGEM_RAFAEL_2020_1-after_drop_feature_by_correlation-02-11-2022_18:29:05.csv'.format(folder_path)
    }

    if sampling not in csv_files.keys():
        raise ValueError(
            'The sampling parameter must be one of values: {}.'.format(csv_files.keys()))

    return csv_files.get(sampling)
