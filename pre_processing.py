import reports
import utils

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, OrdinalEncoder, LabelBinarizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


@utils.timeit
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
            print('Linhas duplicadas deletadas.')
        else:
            print('Nenhuma linha duplicada encontrada para o atributo {}.'.format(
                attribute_name))
    else:
        print(
            '!!!>>> Atributo {} não encontrado no DataFrame para delete duplicates rows by attibute.'.format(attribute_name))
    print('*****FIM DELETE DUPLICATE ROWS BY ATTRIBUTE*********')
    return data_frame


@utils.timeit
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


@utils.timeit
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
                    data_frame[[column]]).astype('uint8')
                columns_ordinal_encoded[column] = encoder_column
            else:
                print('!!!>>> A coluna {} já está codificada.'.format(column))
        else:
            print(
                '!!!>>> Coluna {} não encontrada no DataFrame para ordinal encoding.'.format(column))
    print('*****FIM ORDINAL ENCODER*********')
    return data_frame, columns_ordinal_encoded


@utils.timeit
def inverse_ordinal_encoder_columns(data_frame: pd.DataFrame, columns_ordinal_encoded: dict) -> tuple:
    """Inverse ordinal encode the DataFrame, given by the dictionary containing the coded columns, passed as parameters.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        columns_ordinal_encoded (dict): Dictionary with the ordinal encoder for each column.

    Returns:
        tuple: A DataFrame with the inverse ordinal encoded, and the columns_ordinal_encoded cleared.
    """
    print('\n*****INICIO INVERSE ORDINAL ENCODER******')
    for column, encoder_column in columns_ordinal_encoded.copy().items():
        if column in data_frame.columns:
            data_frame[column] = encoder_column.inverse_transform(
                data_frame[[column]])
            columns_ordinal_encoded.pop(column)
        else:
            print(
                '!!!>>> Coluna {} não encontrada no DataFrame para inverse ordinal encoding.'.format(column))
    print('*****FIM INVERSE ORDINAL ENCODER*********')
    return data_frame, columns_ordinal_encoded


@utils.timeit
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
                    data_frame[column]).astype('uint8')
                columns_label_encoded[column] = encoder_column
            else:
                print('!!!>>> A coluna {} já está codificada.'.format(column))
        else:
            print(
                '!!!>>> Coluna {} não encontrada no DataFrame para label encoding.'.format(column))
    print('*****FIM LABEL ENCODER*********')
    return data_frame, columns_label_encoded


@utils.timeit
def inverse_label_encoder_columns(data_frame: pd.DataFrame, columns_label_encoded: dict) -> tuple:
    """Inverse label encode the DataFrame, given by the dictionary containing the coded columns, passed as parameters.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        columns_label_encoded (dict): Dictionary with the label encoded for each column.

    Returns:
        tuple: A DataFrame with the inverse label encoded, and the columns_label_encoded cleared.
    """
    print('\n*****INICIO INVERSE LABEL ENCODER******')
    for column, encoder_column in columns_label_encoded.copy().items():
        if column in data_frame.columns:
            data_frame[column] = encoder_column.inverse_transform(
                data_frame[column])
            columns_label_encoded.pop(column)
        else:
            print(
                '!!!>>> Coluna {} não encontrada no DataFrame para inverse label encoding.'.format(column))
    print('*****FIM INVERSE LABEL ENCODER*********')
    return data_frame, columns_label_encoded


@utils.timeit
def label_binarizer_column(data_frame: pd.DataFrame, columns_label_binarized: dict, column_name: str) -> tuple:
    """Label binarizer the column, given the column name passed as parameters.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        columns_label_binarized (dict): Dictionary with the label binarizers for each column.
        column_name (str): Column name to label binarizer.

    Returns:
        tuple: A tuple with the data_frame, encoded column, and the columns_label_binarized, in this order.
    """
    print('\n*****INICIO LABEL BINARIZER******')
    if column_name in data_frame.columns:
        if column_name not in columns_label_binarized:
            encoder_column = LabelBinarizer()
            encoded_column = encoder_column.fit_transform(
                data_frame[column_name]).astype('uint8')
            columns_label_binarized[column_name] = encoder_column
            data_frame = utils.delete_columns(
                data_frame=data_frame, delete_columns_names=[column_name])
        else:
            print('!!!>>> A coluna {} já está codificada.'.format(column_name))
    else:
        print(
            '!!!>>> Coluna {} não encontrada no DataFrame para label binarizer.'.format(column_name))
    print('*****FIM LABEL BINARIZER*********')
    return data_frame, encoded_column, columns_label_binarized


@utils.timeit
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
                encoded_df = pd.DataFrame(
                    encoder_column.fit_transform(data_frame[[column]]),
                    columns=encoder_column.get_feature_names_out(),
                    index=data_frame.index.values.tolist(), dtype='uint8'
                )
                data_frame = pd.concat([data_frame, encoded_df], axis=1)
                data_frame = utils.delete_columns(
                    data_frame=data_frame, delete_columns_names=[column])
                columns_one_hot_encoded[column] = encoder_column
            else:
                print('!!!>>> A coluna {} já está codificada.'.format(column))
        else:
            print(
                '!!!>>> Coluna {} não encontrada no DataFrame para one-hot encoding.'.format(column))
    print('*****FIM ONE-HOT ENCODER*********')
    return data_frame, columns_one_hot_encoded


@utils.timeit
def inverse_one_hot_encoder_columns(data_frame: pd.DataFrame, columns_one_hot_encoded: dict) -> tuple:
    """Inverse one-hot encode the DataFrame, given by the dictionary containing the coded columns, passed as parameters.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        columns_one_hot_encoded (dict): Dictionary with the one-hot encoded for each column.

    Returns:
        tuple: A DataFrame with the inverse one-hot encoded, and the columns_one_hot_encoded cleared.
    """
    print('\n*****INICIO INVERSE ONE-HOT ENCODER******')
    for column, encoder_column in columns_one_hot_encoded.copy().items():
        if len(set(encoder_column.get_feature_names_out()).intersection(set(data_frame.columns))) == len(encoder_column.get_feature_names_out()):
            data_frame[column] = encoder_column.inverse_transform(
                data_frame[encoder_column.get_feature_names_out()])
            data_frame = utils.delete_columns(
                data_frame=data_frame, delete_columns_names=encoder_column.get_feature_names_out())
            columns_one_hot_encoded.pop(column)
        else:
            print(
                '!!!>>> As  colunas {} não foram encontradas todas no DataFrame para inverse one-hot encoding.'.format(encoder_column.get_feature_names_out()))
    print('*****FIM INVERSE ONE-HOT ENCODER*********')
    return data_frame, columns_one_hot_encoded


@utils.timeit
def drop_feature_by_correlation(data_frame: pd.DataFrame, method: str, columns_names: list, threshold: float = 0.95) -> pd.DataFrame:
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
            correlation_matrix = data_frame.corr(
                method=method).astype('float32')
            correlation_matrix = correlation_matrix.loc[column, :]
            correlation_matrix = correlation_matrix.sort_values(
                ascending=False)
            correlation_matrix = correlation_matrix[correlation_matrix > threshold]
            correlation_matrix = correlation_matrix.drop(column)
            correlation_matrix = correlation_matrix.dropna()
            correlation_matrix = correlation_matrix.index.tolist()
            if len(correlation_matrix) > 0:
                print('>>> Correlação, para o limite de {}, entre o atributo {} e o(s) atributo(s) {} foi encontrada.\n>>>>Removendo o(s) atributo(s) do dataframe.'.format(
                    threshold, column, correlation_matrix))
                data_frame.drop(columns=correlation_matrix, inplace=True)
            else:
                print(
                    '>>> Nenhuma correlação, para o limite de {}, encontrada para o atributo {}.'.format(threshold, column))
        else:
            print(
                '!!!>>> Coluna {} não encontrada no DataFrame para drop feature by correlation.'.format(column))
    print('*****FIM DROP FEATURE BY CORRELATION*********')
    return data_frame


@utils.timeit
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


@utils.timeit
def inverse_min_max_scaler_columns(data_frame: pd.DataFrame, columns_min_max_scaled: dict) -> tuple:
    """Inverse Min-Max scale the DataFrame, given by the dictionary containing the scaled columns, passed as parameters.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        columns_min_max_scaled (dict): Dictionary with the min-max scaled for each column.

    Returns:
        tuple: A DataFrame with the inverse min max scaled, and the columns_min_max_scaled cleared.
    """
    print('\n*****INICIO INVERSE MIN-MAX SCALER******')
    for column, scaler_column in columns_min_max_scaled.copy().items():
        if column in data_frame.columns:
            data_frame[column] = scaler_column.inverse_transform(
                data_frame[column].values.reshape(-1, 1))
            columns_min_max_scaled.pop(column)
        else:
            print(
                '!!!>>> Coluna {} não encontrada no DataFrame para inverse min-max scaler.'.format(column))
    print('*****FIM INVERSE MIN-MAX SCALER*********')
    return data_frame, columns_min_max_scaled


@utils.timeit
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


@utils.timeit
def delete_columns_with_single_value(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Delete the columns with single value.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.

    Returns:
        pd.DataFrame: A DataFrame with the columns with single value deleted.
    """
    print('\n*****INICIO DELETE COLUMNS WITH SINGLE VALUE******')
    counts = data_frame.nunique()

    to_del = [i for i, v in counts.items() if v == 1]

    if to_del:
        data_frame.drop(to_del, axis=1, inplace=True)
        print('>>> Colunas removidas com valor único: {}'.format(to_del))
    else:
        print('>>> Nenhuma coluna com valor único encontrada.')

    print('*****FIM DELETE COLUMNS WITH SINGLE VALUE*********')
    return data_frame


@utils.timeit
def delete_columns_with_low_variance(x: pd.DataFrame, threshold: float = 0.0, separate_numeric_columns: bool = False) -> pd.DataFrame:
    """Delete the columns with low variance.

    Args:
        x (pd.DataFrame): Pandas DataFrame with numeric data to be treated.
        threshold (float, optional): Threshold of variance. Defaults to 0.8.
        separate_numeric_columns (bool, optional): Separate the numeric columns. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame with the columns with low variance deleted.
    """
    print('\n*****INICIO DELETE COLUMNS WITH LOW VARIANCE******')
    print('>>> Número de colunas antes da remoção: {}'.format(x.shape[1]))
    transform = VarianceThreshold(threshold=threshold)
    if separate_numeric_columns:
        x, x_numeric = utils.separate_numeric_dataframe_columns(
            x=x, exclude_types=utils.TYPES_EXCLUDE_DF)
        transform.fit_transform(x_numeric)
        x_numeric = utils.delete_columns(data_frame=x_numeric, delete_columns_names=list(
            set(transform.feature_names_in_) -
            set(transform.get_feature_names_out())
        ))
        x = utils.concatenate_data_frames(data_frames=[x, x_numeric])
    else:
        transform.fit_transform(x)
        x = utils.delete_columns(data_frame=x, delete_columns_names=list(
            set(transform.feature_names_in_) -
            set(transform.get_feature_names_out())
        ))
    print('>>> Threshold: {:.2f}\nNúmero de colunas depois da remoção: {}'.format(
        threshold, x.shape[1]))
    print('*****FIM DELETE COLUMNS WITH LOW VARIANCE*********')
    return x


@utils.timeit
def select_features_from_model(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, model: str, max_features: int = None, threshold: float = None) -> tuple:
    """Select the features from a model.
    SelectFromModel, from scikit-learn, is a meta-transformer that can be used alongside any estimator that assigns importance to each feature through a specific attribute (such as coef_, feature_importances_) or via an importance_getter callable after fitting.
    The models supported for this function are: DecisionTreeClassifier (decision_tree_classifier), RandomForestClassifier (random_forest_classifier), XGBClassifier (xgb_classifier), LogisticRegression (logistic_regression), LinearSVC (linear_svc), SGDClassifier (sgd_classifier).

    Args:
        x_train (np.ndarray): X train data.
        y_train (np.ndarray): Y train data.
        x_test (np.ndarray): X test data.
        model (str): Model to be used as estimator. For example: 'decision_tree_classifier'.
        max_features (int, optional): Maximum number of features to select. Defaults to None.
        threshold (float, optional): The threshold value to use for feature selection. Defaults to None.

    Returns:
        tuple: A tuple with the selected features for the x_train e x_test, and the select from model object adjusted for data.
    """
    print('\n*****INICIO SELECT FEATURES FROM MODEL******')
    if model == 'decision_tree_classifier':
        model = DecisionTreeClassifier()
    elif model == 'random_forest_classifier':
        model = RandomForestClassifier()
    elif model == 'xgb_classifier':
        model = XGBClassifier()
    elif model == 'logistic_regression':
        model = LogisticRegression()
    elif model == 'linear_svc':
        model = LinearSVC()
    elif model == 'sgd_classifier':
        model = SGDClassifier()
    else:
        raise Exception('Model not supported.')

    print('>>> Número de colunas antes da seleção. x_train: {}. x_test: {}'.format(
        x_train.shape[1], x_test.shape[1]))

    select_from_model = SelectFromModel(
        model, max_features=max_features, threshold=threshold)

    select_from_model.fit(x_train, y_train)

    x_train_fs = select_from_model.transform(x_train)
    x_test_fs = select_from_model.transform(x_test)

    print('>>> Número de colunas depois da seleção. x_train_fs: {}. x_test_fs: {}'.format(
        x_train_fs.shape[1], x_test_fs.shape[1]))

    print('*****FIM SELECT FEATURES FROM MODEL*********')
    return x_train_fs, x_test_fs, select_from_model


def create_ordinal_encoder_transformer(ordinal_encoder_columns_names: dict, data_frame_columns: list, dtype: type = np.uint8, with_categories: bool = True, handle_unknown: str = 'error', imputer: object = None, unknown_value=None) -> tuple:
    """Create an ordinal encoder transformer, for ColumnTransformer used in pipeline.
    More in: https://scikit-learn.org/stable/modules/compose.html

    Args:
        ordinal_encoder_columns_names (dict): Dictionary with the columns names to be encoded and the categories.
        data_frame_columns (list): Columns of the data frame.
        dtype (type, optional): Data type of the encoded columns. Defaults to np.uint8.
        with_categories (bool, optional): If True, the object OrdinalEncoder, with categories instancied, will be returned. Defaults to True.
        handle_unknown (str, optional): When set to 'error' an error will be raised in case an unknown categorical feature is present during transform. When set to 'use_encoded_value', the encoded value of unknown categories will be set to the value given for the parameter unknown_value. Defaults to 'error'.
        imputer (object, optional): Imputer object to be used in the pipeline. Defaults to None.
        unknown_value (optional): Value to use for unknown categories. If handle_unknown is 'use_encoded_value', the value should be int or np.nan. See documentation for OrdinalEncoder for more: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html. Defaults to None.

    Returns:
        tuple: A tuple with name for transformer, OrdinalEncoder parametrized, and columns to be encoded.
    """
    if not ordinal_encoder_columns_names:
        raise Exception(
            'ordinal_encoder_columns_names must be informed, for create ordinal encoder transformer.')

    columns_with_categories = dict()
    columns_without_categories = list()

    for column, categories in ordinal_encoder_columns_names.items():
        if column not in data_frame_columns:
            print(
                '!!!> Column {} not in dataframe columns. The column will not be considered in the ordinal encoder.'.format(column))
            continue

        if categories is None:
            columns_without_categories.append(column)
        else:
            columns_with_categories.update({column: categories})

    ordinal_encoder_transformer = tuple()
    name_transformer = str()
    transformer = None
    columns_to_transformer = list()

    if with_categories:
        if columns_with_categories:
            name_transformer = 'ordinal_encoder_with_categories'

            if imputer:
                transformer = Pipeline(
                    steps=[
                        ('imputer', imputer),
                        ('ordinal_encoder', OrdinalEncoder(
                            categories=list(columns_with_categories.values()),
                            dtype=dtype,
                            handle_unknown=handle_unknown,
                            unknown_value=unknown_value
                        ))
                    ]
                )
            else:
                transformer = OrdinalEncoder(
                    categories=list(columns_with_categories.values()),
                    dtype=dtype,
                    handle_unknown=handle_unknown,
                    unknown_value=unknown_value
                )

            columns_to_transformer = list(columns_with_categories.keys())
        else:
            raise Exception(
                '!!!> No columns with categories informed for ordinal encoder transformer.')
    else:
        if columns_without_categories:
            name_transformer = 'ordinal_encoder_without_categories'

            if imputer:
                transformer = Pipeline(
                    steps=[
                        ('imputer', imputer),
                        ('ordinal_encoder', OrdinalEncoder(
                            dtype=dtype,
                            handle_unknown=handle_unknown,
                            unknown_value=unknown_value
                        ))
                    ]
                )
            else:
                transformer = OrdinalEncoder(
                    dtype=dtype,
                    handle_unknown=handle_unknown,
                    unknown_value=unknown_value
                )

            columns_to_transformer = columns_without_categories
        else:
            raise Exception(
                '!!!> No columns without categories informed for ordinal encoder transformer.')

    ordinal_encoder_transformer = tuple(
        (
            name_transformer,
            transformer,
            columns_to_transformer
        )
    )

    return ordinal_encoder_transformer


def create_one_hot_encoder_transformer(columns: list, data_frame_columns: list, sparse: bool = False, handle_unknown: str = 'infrequent_if_exist', drop: str = 'if_binary', dtype: type = np.uint8, imputer: object = None) -> tuple:
    """Create a one hot encoder transformer, for ColumnTransformer used in pipeline.
    More in: https://scikit-learn.org/stable/modules/compose.html

    Args:
        columns (list): Columns to be encoded.
        data_frame_columns (list): Columns of the data frame.
        sparse (bool, optional): If True, a sparse matrix will be returned. Defaults to False.
        handle_unknown (str, optional): Whether to raise an error or ignore if an unknown categorical feature is present during transform (default is to raise). Defaults to 'infrequent_if_exist'.
        drop (str, optional): Specifies a methodology to use to drop one of the categories per feature. Defaults to 'if_binary'.
        dtype (type, optional): Data type of the encoded columns. Defaults to np.uint8.
        imputer (object, optional): Imputer object to be used in the pipeline. Defaults to None.

    Returns:
        tuple: A tuple with name for transformer, OneHotEncoder parametrized, and columns to be encoded.
    """
    if not columns:
        raise Exception(
            'Columns must be informed, for create one hot encoder transformer.')

    for column in columns:
        if column not in data_frame_columns:
            print(
                '!!!> Column {} not in dataframe. The column will not be considered in the one hot encoder.'.format(column))
            columns.remove(column)

    one_hot = OneHotEncoder(
        sparse=sparse, handle_unknown=handle_unknown, drop=drop, dtype=dtype)

    transformer = None
    if imputer:
        transformer = Pipeline(
            steps=[
                ('imputer', imputer),
                ('one_hot', one_hot)
            ]
        )
    else:
        transformer = one_hot
    return ('one_hot_encoder', transformer, columns)


def create_min_max_scaler_transformer(columns: list, data_frame_columns: list, imputer: object = None) -> tuple:
    """Create a min max scaler transformer, for ColumnTransformer used in pipeline.
    More in: https://scikit-learn.org/stable/modules/compose.html

    Args:
        columns (list): Columns to be scaled.
        data_frame_columns (list): Columns of the data frame.
        imputer (object, optional): Imputer object to be used in the pipeline. Defaults to None.

    Returns:
        tuple: A tuple with name for transformer, MinMaxScaler parametrized, and columns to be scaled.
    """
    if not columns:
        raise Exception(
            'Columns must be informed, for create min max scaler transformer.')

    for column in columns:
        if column not in data_frame_columns:
            print(
                '!!!> Column {} not in dataframe. The column will not be considered in the min max scaler.'.format(column))
            columns.remove(column)

    min_max_scaler = MinMaxScaler()

    transformer = None
    if imputer:
        transformer = Pipeline(
            steps=[
                ('imputer', imputer),
                ('min_max', min_max_scaler)
            ]
        )
    else:
        transformer = min_max_scaler

    return ('min_max_scaler', transformer, columns)


@utils.timeit
def simple_imputer_dataframe(data_frame: pd.DataFrame, columns: list, strategy: str = 'mean', fill_value: str = None, verbose: bool = False) -> pd.DataFrame:
    """Simple imputer for data frame.

    Args:
        data_frame (pd.DataFrame): Data frame to be imputed.
        columns (list): Columns to be imputed.
        strategy (str, optional): Strategy to impute. Defaults to 'mean'.
        fill_value (str, optional): If strategy is constant, fill_value is used to replace all occurrences of missing_values. Defaults to None.
        verbose (bool, optional): If True, print the columns with missing values. Defaults to False.

    Returns:
        pd.DataFrame: Data frame with missing values imputed.
    """
    if not columns:
        raise Exception('Columns must be informed, for simple impute.')

    for column in columns:
        if column not in data_frame.columns:
            print(
                '!!!> Column {} not in dataframe. The column will not be considered in the imputer.'.format(column))
            columns.remove(column)

    imputer = instance_simple_imputer(
        missing_values=np.nan, strategy=strategy, fill_value=fill_value)
    imputer.fit(data_frame[columns])

    if verbose:
        print('Statistics: {}'.format(
            imputer.statistics_))

    data_frame[columns] = imputer.transform(data_frame[columns])

    return data_frame


def create_simple_imputer_transformer(columns: list, data_frame_columns: list, strategy: str = 'mean', fill_value: str = None) -> tuple:
    """Create a simple imputer transformer, for ColumnTransformer used in pipeline.
    More in: https://scikit-learn.org/stable/modules/compose.html

    Args:
        columns (list): Columns to be imputed.
        data_frame_columns (list): Columns of the data frame.
        strategy (str, optional): Strategy to impute. Defaults to 'mean'.
        fill_value (str, optional): If strategy is constant, fill_value is used to replace all occurrences of missing_values. Defaults to None.

    Returns:
        tuple: A tuple with name for transformer, SimpleImputer parametrized, and columns to be imputed.
    """
    if not columns:
        raise Exception(
            'Columns must be informed, for create simple imputer transformer.')

    for column in columns:
        if column not in data_frame_columns:
            print(
                '!!!> Column {} not in dataframe. The column will not be considered in the simple imputer.'.format(column))
            columns.remove(column)

    simple_imputer = instance_simple_imputer(
        missing_values=np.nan, strategy=strategy, fill_value=fill_value)
    return ('simple_imputer', simple_imputer, columns)


def instance_simple_imputer(missing_values: object = np.nan, strategy: str = 'mean', fill_value: str = None) -> SimpleImputer:
    """Instance a simple imputer.

    Args:
        missing_values (object, optional): The placeholder for the missing values. All occurrences of missing_values will be imputed. For missing values encoded as np.nan, use the string value "nan". Defaults to np.nan.
        strategy (str, optional): The imputation strategy. Defaults to 'mean'.
        fill_value (str, optional): If strategy is constant, fill_value is used to replace all occurrences of missing_values. Defaults to None.

    Returns:
        SimpleImputer: Simple imputer instance.
    """
    return SimpleImputer(missing_values=missing_values, strategy=strategy, fill_value=fill_value)
