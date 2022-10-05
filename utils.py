import numpy as np
import pandas as pd
from datetime import datetime

# Types of columns to be excluded from the DataFrame
TYPES_EXCLUDE_DF = [pd.CategoricalDtype, pd.DatetimeTZDtype, np.datetime64]


def get_current_datetime() -> str:
    """Get the current datetime.

    Returns:
        str: Current datetime.
    """
    return datetime.now().strftime('%d-%m-%Y_%H:%M:%S')


def separate_numeric_numpy_columns(x: np.array) -> tuple:
    """Separate the non-numeric columns from the numeric columns.

    Args:
        x (np.array): Numpy array to be treated.

    Returns:
        tuple: Tuple with the non-numeric columns (x) and the numeric columns (x_numeric).
    """
    print('X shape (Dados originais): {}'.format(x.shape))
    x_numeric = np.empty(shape=[x.shape[0], 0])
    for i in range(0, x.shape[1]):
        try:
            if isinstance(x[:, i][0], (int, float)):
                x_numeric = np.append(
                    x_numeric, x[:, i].reshape(-1, 1), axis=1)
                x = np.delete(x, i, axis=1)
        except IndexError:
            break
    print('Após separar os atributos numéricos.')
    print('X shape (Dados não numéricos): {}'.format(x.shape))
    print('X_numeric shape (Dados numéricos): {}'.format(x_numeric.shape))
    return x, x_numeric


def separate_numeric_dataframe_columns(x: pd.DataFrame, exclude_types: list) -> tuple:
    """Separate the non-numeric columns from the numeric columns.

    Args:
        x (pd.DataFrame): DataFrame to be treated.
        exclude_types (list): List of types to be excluded from the DataFrame (x).

    Returns:
        tuple: Tuple with the non-numeric columns (x) and the numeric columns (x_numeric).
    """
    print('X shape (Dados originais): {}'.format(x.shape))
    x_numeric = x.select_dtypes(exclude=exclude_types)
    x = delete_columns(
        data_frame=x, delete_columns_names=x_numeric.columns.to_list())
    print('Após separar os atributos numéricos.')
    print('X shape (Dados não numéricos): {}'.format(x.shape))
    print('X_numeric shape (Dados numéricos): {}'.format(x_numeric.shape))
    return x, x_numeric


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


def create_x_y_numpy_data(data_frame: pd.DataFrame, print_memory_usage: bool = False) -> tuple:
    """Create x and y data from a DataFrame.

    Args:
        data_frame (pd.DataFrame): DataFrame with last cloumn as a targe class (y).
        print_memory_usage (bool, optional): If True, print the memory usage of x and y. Defaults to False.

    Returns:
        tuple: Return x and y data from the DataFrame, being x and y of type numpy.array.
    """
    x = np.array(data_frame)[:, :-1]
    y = np.array(data_frame)[:, -1]
    if print_memory_usage:
        print('Data frame memory usage: {} Bytes'.format(
            data_frame.memory_usage(index=True, deep=True).sum()))
        print('X memory usage: {} Bytes'.format(x.nbytes))
        print('Y memory usage: {} Bytes'.format(y.nbytes))
    return x, y


def create_x_y_dataframe_data(data_frame: pd.DataFrame, print_memory_usage: bool = False) -> tuple:
    """Create x and y data from a DataFrame.

    Args:
        data_frame (pd.DataFrame): DataFrame with last cloumn as a targe class (y).
        print_memory_usage (bool, optional): If True, print the memory usage of x and y. Defaults to False.

    Returns:
        tuple: Return x and y data from the DataFrame, being x of a DataFrame type, and y of a Series type.
    """
    x = data_frame.iloc[:, :-1]
    y = data_frame.iloc[:, -1]
    if print_memory_usage:
        print('X memory usage: {} Bytes'.format(
            x.memory_usage(index=True, deep=True).sum()))
        print('Y memory usage: {} Bytes'.format(
            y.memory_usage(index=True, deep=True)))
    return x, y


def concatenate_data_frames(data_frames: list) -> pd.DataFrame:
    """Concatenate a list of DataFrames.

    Args:
        data_frames (list): List of DataFrames to be concatenated.

    Returns:
        pd.DataFrame: Concatenated DataFrame.
    """
    data_frame = pd.concat(data_frames, axis=1)
    return data_frame
