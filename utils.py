import numpy as np
import pandas as pd
from datetime import datetime


def get_current_datetime() -> str:
    """Get the current datetime.

    Returns:
        str: Current datetime.
    """
    return datetime.now().strftime('%d-%m-%Y_%H:%M:%S')


def separate_numeric_columns(x: np.array) -> tuple:
    """Separate the non-numeric columns from the numeric columns.

    Args:
        data_frame (np.array): Numpy array to be treated.

    Returns:
        tuple: Tuple with the non-numeric columns (x) and the numeric columns (x_aux).
    """
    print('X shape (Dados originais): {}'.format(x.shape))
    x_aux = np.empty(shape=[x.shape[0], 0])
    for i in range(0, x.shape[1]):
        try:
            if isinstance(x[:, i][0], (int, float)):
                x_aux = np.append(x_aux, x[:, i].reshape(-1, 1), axis=1)
                # print(type(x_aux[:, x_aux.shape[1]-1][0]), x_aux[:, x_aux.shape[1]-1][0])
                x = np.delete(x, i, axis=1)
        except IndexError:
            break
    print('Após separar os atributos numéricos.')
    print('X shape (Dados não numéricos): {}'.format(x.shape))
    print('X_aux shape (Dados numéricos): {}'.format(x_aux.shape))
    return x, x_aux


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
