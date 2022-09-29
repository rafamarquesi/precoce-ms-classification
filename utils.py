import numpy as np
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
