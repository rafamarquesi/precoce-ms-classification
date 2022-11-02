from functools import wraps
import time
from typing import Callable
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Types of columns to be excluded from the DataFrame
TYPES_EXCLUDE_DF = [pd.CategoricalDtype, pd.DatetimeTZDtype, np.datetime64]

# Pandas max rows, None displays all rows
PANDAS_MAX_ROWS = 5000


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


def convert_pandas_dtype_to_numpy_dtype(data_frame: pd.DataFrame, pandas_dtypes: list) -> pd.DataFrame:
    """Convert pandas dtype to numpy dtype.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        pandas_dtypes (list): List of pandas dtypes to be converted to numpy dtypes.

    Returns:
        pd.DataFrame: A DataFrame with the columns converted to numpy dtype.
    """
    if pandas_dtypes:
        for column in data_frame.columns:
            if data_frame[column].dtype in pandas_dtypes:
                data_frame[column] = data_frame[column].astype(
                    data_frame[column].dtype.type)
    else:
        raise Exception('pandas_dtypes not informed.')
    return data_frame


def random_sampling_data(data_frame: pd.DataFrame, how_generate: str, n: int = 0, frac: float = 0.0, rate: int = 0) -> pd.DataFrame:
    """Random sampling data.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        how_generate (str): How generate the random sampling data. Options: 'exact_number', 'percentage', and 'constant_rate'.
        n (int, optional): Number of samples to be generated, in option exact number. Defaults to 0.
        frac (float, optional): Fraction of samples to be generated, in option percentage. Defaults to 0.0.
        rate (int, optional): Constant rate of samples to be generated, in option constant rate. Defaults to 0.

    Returns:
        pd.DataFrame: A DataFrame with the data sampled.
    """
    if how_generate == 'exact_number' and n > 0:
        print('Random sampling data with exact number.')
        return data_frame.sample(n=n)
    elif how_generate == 'percentage' and (frac > 0.0 and frac <= 1.0):
        print('Random sampling data with percentage.')
        return data_frame.sample(frac=frac)
    elif how_generate == 'constant_rate' and rate > 0:
        print('Random sampling data with constante rate.')
        return data_frame[::rate]
    else:
        raise Exception('Invalid option for generate samples.')


def convert_seconds_to_time(seconds: float) -> str:
    """Convert seconds to time.

    Args:
        seconds (float): Number of seconds.

    Returns:
        str: Time in format HH:MM:SS.
    """
    return str(timedelta(seconds=seconds))


#################### DECORATORS ####################
def timeit(func: Callable) -> Callable:
    """Decorator to measure the time of execution of a function.

    Args:
        func (Callable): Function to be measured.

    Returns:
        Callable: Function with the time of execution.
    """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs) -> Callable:
        """Measure the time of execution of a function.

        Returns:
            Callable: The function with their time of execution.
        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time

        print(
            'Function {} Took {}\n'.format(
                func.__name__,
                convert_seconds_to_time(seconds=total_time)
            )
        )
        return result
    return timeit_wrapper
