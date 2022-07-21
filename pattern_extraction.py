import pandas as pd
import numpy as np


def create_x_y_data(data_frame: pd.DataFrame) -> tuple:
    """Create x and y data from a DataFrame.

    Args:
        data_frame (pd.DataFrame): DataFrame with last cloumn as a targe class (y).

    Returns:
        tuple: Return x and y data from the DataFrame, being x and y of type numpy.ndarray.
    """
    x = np.array(data_frame)[:, :-1]
    y = np.array(data_frame)[:, -1]
    return x, y
