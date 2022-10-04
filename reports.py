import random
import utils
import pattern_extraction
import pre_processing
from sys import displayhook
import pandas as pd
import numpy as np
import seaborn as sns
import dataframe_image as dfi
import matplotlib.pyplot as plt


def informations(data_frame: pd.DataFrame) -> None:
    """Print some informations of the DataFrame.

    Args:
        data_frame (pd.DataFrame): DataFrame to be printed.
    """

    print('\n*****INICIO PRINT INFOS******')
    # self.relatorio_atributos_base_dados(dados=dados_temp)
    print('Número total de linhas do DataFrame: {}'.format(len(data_frame.index)))
    print('Número de colunas: {}'.format(data_frame.columns.size))
    print('Informações do DataFrame:')
    data_frame.info(verbose=True, memory_usage='deep')
    print('*****FIM PRINT INFOS*********')


def print_list_columns(data_frame: pd.DataFrame) -> None:
    """Show the columns of the DataFrame.

    Args:
        data_frame (pd.DataFrame): DataFrame to be printed.
    """
    print('\n*****INICIO SHOW COLUNAS******')
    print(list(data_frame.columns))
    print('*****FIM SHOW COLUNAS*********')


def nan_attributes(data_frame: pd.DataFrame, total_nan: int) -> None:
    """Report of attributes with nan values.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        total_nan (int): Number of nan values.
    """
    print('\n*****INICIO RELATÓRIO ATRIBUTOS NAN******')
    print('Total de Ocorrencia de NaN no DataFrame: {}'.format(total_nan))
    print('Linhas com valores NaN: {}'.format(
        data_frame[data_frame.isna().sum(axis=1) > 0]))
    print('Colunas com Ocorrencia de NaN no DataFrame: {}'.format(
        data_frame.columns[data_frame.isna().any()].tolist()))
    print('----Colunas sumarizado o total de ocorrência de NaN')
    df_nan = pd.DataFrame(columns=['Coluna', 'total_NaN'])
    for column in data_frame.columns[data_frame.isna().any()].tolist():
        df_nan = pd.concat([df_nan, pd.DataFrame.from_records(
            [
                {
                    'Coluna': column,
                    'total_NaN': data_frame[column].isna().sum(),
                    'Porcentagem': data_frame[column].isna().sum() / len(data_frame[column]) * 100
                }
            ])])
        # df_nan = df_nan.append(
        #     {'Coluna': column, 'total_NaN': data_frame[column].isna().sum()}, ignore_index=True)
    df_nan = df_nan.sort_values(['total_NaN'], ascending=[False])
    print(df_nan)
    print('*****FIM RELATÓRIO ATRIBUTOS NAN******')


def duplicate_rows_by_attribute(data_frame: pd.DataFrame, rows_duplicated: pd.DataFrame, attribute: str) -> None:
    """Report of duplicate rows by attribute.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        rows_duplicated (pd.DataFrame): DataFrame with duplicate rows.
        attribute (str): Attribute to be used to report.
    """
    print('\n*****INICIO RELATÓRIO LINHAS DUPLICADAS******')
    print('Data Frame do Atríbuto {} com as linhas duplicadas:\n'.format(
        attribute))
    displayhook(rows_duplicated)
    print('Relatório dos atributos que divergem, entre os registros que tem o atributo {} igual.'.format(attribute))
    for id in rows_duplicated[attribute].unique():
        print('{}:{}'.format(attribute, id))
        rows_duplicated = data_frame.loc[data_frame[attribute] == id]
        for column_name, column_data in rows_duplicated.iteritems():
            comparison = column_data.ne(
                column_data.shift().bfill().astype(column_data.dtype.name)).astype('uint8').values
            if not np.all(comparison == comparison[0]):
                print('Nome coluna que diverge: {}'.format(column_name))
                print('Index das linhas e valor na coluna que diverge:\n{}'.format(
                    column_data))
                # print(column_data.ne(column_data.shift().bfill()).astype('uint8'))
                print('-------------------------------')
        print('Próximo ++++++++++++++')
    print('*****FIM RELATÓRIO LINHAS DUPLICADAS******')


def all_attributes(data_frame: pd.DataFrame) -> None:
    """Report of all attributes.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
    """
    print('\n*****INICIO RELATÓRIO ATRIBUTOS******')
    print('Número de atributos: {}'.format(len(data_frame.columns)))
    print('Nome dos atributos: {}'.format(list(data_frame.columns)))
    for column_name, column_data in data_frame.iteritems():
        print('Nome da coluna: {}'.format(column_name))
        print(column_data.value_counts())
        print('-------------------------------')
    print('*****FIM RELATÓRIO ATRIBUTOS******')


def class_distribution(y: np.array) -> None:
    """Report of class distribution.

    Args:
        y (np.array): NumPy Array to be treated.
    """
    print('\n*****INICIO RELATÓRIO DISTRIBUIÇÃO DE CLASSES******')
    dist_1 = y[y.nonzero()].size / y.size
    dist_0 = (y.size - y[y.nonzero()].size) / y.size
    print('Distribuição da classe 1: {0:.0%}'.format(dist_1))
    print('Distribuição da classe 0: {0:.0%}'.format(dist_0))
    if dist_1 > dist_0:
        print('Erro majoritário: {0:.0%}'.format(1 - dist_1))
    else:
        print('Erro majoritário: {0:.0%}'.format(1 - dist_0))
    print('*****FIM RELATÓRIO DISTRIBUIÇÃO DE CLASSES******')


def correlation_matrix(data_frame: pd.DataFrame, method: str, attribute: str = None, display_matrix: bool = False, export_matrix: bool = False, path_save_matrix: str = None) -> None:
    """Create a correlation matrix from the DataFrame.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        method (str): Method to be used to create the correlation matrix.
        attribute (str, optional): Attribute to be used to create the correlation matrix. Defaults to None.
        display_matrix (bool, optional): Flag to display the correlation matrix. Defaults to False.
        export_matrix (bool, optional): Flag to export the correlation matrix. Defaults to False.
        path_save_matrix (str, optional): Path to save the correlation matrix. Defaults to None.
    """
    print('\n*****INICIO CORRELATION MATRIX******')
    if attribute is None:
        correlation_matrix = data_frame.corr(method=method).astype('float32')

        cmap = sns.diverging_palette(5, 250, as_cmap=True)

        styled_table = correlation_matrix.style.background_gradient(cmap, axis=1)\
            .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
            .set_caption("Hover to magify")\
            .set_precision(2)\
            .set_table_styles(__magnify())
    else:
        correlation_matrix = data_frame.corr(
            method=method).astype('float32')[attribute]
        styled_table = correlation_matrix.to_frame().style.background_gradient(
            cmap=sns.light_palette((260, 75, 60), input="husl", as_cmap=True))

    if display_matrix:
        displayhook(styled_table)

    if export_matrix:
        if path_save_matrix is None:
            path_save_matrix = 'correlation_matrix-{}.png'.format(
                utils.get_current_datetime())
        else:
            path_save_matrix = path_save_matrix + '/correlation_matrix-{}.png'.format(
                utils.get_current_datetime())

        dfi.export(styled_table, path_save_matrix, max_cols=-1, max_rows=-1)

    print('*****FIM CORRELATION MATRIX*********')


def models_results(models_results: dict, path_save_fig: str = None, display_results: bool = False) -> None:
    """Print the models results.

    Args:
        models_results (dict): Results of the models.
        path_save_fig (str, optional): Path to save the figures. If None, save the figures in root path of project. Defaults to None.
        display_results (bool, optional): Flag to display the results, for example in jupyter notebook. Defaults to False.
    """
    print('\n*****INICIO IMPRIMIR RESULTADOS MODELOS******')
    accuracy_graphic = pd.DataFrame()

    if path_save_fig is None:
        path_save_fig = ''
    else:
        path_save_fig = path_save_fig + '/'

    for key, value in models_results.items():

        cm = sns.light_palette(
            __generate_random_rgb_color(), input="husl", as_cmap=True)

        print('---> Algoritmo: {} \nResultado:\n'.format(key))

        value_styled = value.style.background_gradient(cmap=cm)

        if display_results:
            displayhook(value_styled)
        else:
            print('{}\n\n'.format(value))

        dfi.export(
            value_styled,
            path_save_fig +
            'evaluation-measures-{}-{}'.format(
                key,
                utils.get_current_datetime()
            ),
            max_cols=-1,
            max_rows=-1
        )

        print(
            '\nDescribe:\n{}\n\n\n'.format(value.drop(columns=['Iteração']).describe()))

        accuracy_graphic = pd.concat(
            [
                accuracy_graphic,
                pd.DataFrame(
                    models_results[key].Acurácia.values, columns=[key])
            ],
            axis=1
        )

        fig = plt.figure()
        accuracy_graphic[key].to_frame().boxplot()
        fig.savefig(
            path_save_fig + 'boxplot-accuracy-{}-{}.png'.format(key, utils.get_current_datetime()))

        if display_results:
            plt.show()

    fig = plt.figure()
    accuracy_graphic.boxplot()
    fig.savefig(path_save_fig +
                'boxplot-accuracy-all-models-{}.png'.format(utils.get_current_datetime()))

    if display_results:
        print('\nBoxplot de acurácia de todos os modelos:')
        plt.show()

    print('Desvio padrão da acurácia dos algoritmos:')
    displayhook(accuracy_graphic.std())

    print('\nMédia da acurácia dos algoritmos:')
    displayhook(accuracy_graphic.mean())

    # Ghraphic of mean and standard deviation, of all iterations, of each algorithm
    pd.DataFrame(
        [accuracy_graphic.mean(), accuracy_graphic.std()],
        index=['Média', 'Desvio\nPadrão']
    ).plot.barh().get_figure().savefig(
        path_save_fig +
        'barh-accuracy-meand-std-all-models-{}.png'.format(
            utils.get_current_datetime())
    )

    if display_results:
        plt.show()

    print('*****FIM IMPRIMIR RESULTADOS MODELOS******')


def min_max_column(data_frame: pd.DataFrame) -> None:
    """Print the min and max of the column in the data frame.

    Args:
        data_frame (pd.DataFrame): Data frame to be treated.
    """
    print('\n*****INICIO IMPRIMIR MIN MAX COLUMN******')
    for column in data_frame.columns:
        print('\nColumn:{}\nMin:{}\nMax:{}'.format(
            column, data_frame[column].min(), data_frame[column].max()))
    print('*****FIM IMPRIMIR MIN MAX COLUMN******')


def unique_values_for_each_column(data_frame: pd.DataFrame) -> None:
    """Print the unique values for each column in the data frame.

    Args:
        data_frame (pd.DataFrame): Data frame to be treated.
    """
    print('\n*****INICIO IMPRIMIR UNIQUE VALUES FOR EACH COLUMN******')
    print(data_frame.nunique())
    print('*****FIM IMPRIMIR UNIQUE VALUES FOR EACH COLUMN******')


def percentage_unique_values_for_each_column(data_frame: pd.DataFrame, threshold: float = 100) -> None:
    """Print the percentage of unique values for each column in the data frame.

    Args:
        data_frame (pd.DataFrame): Data frame to be treated.
        threshold (float, optional): If 100, print percentage for all columns. Other threshold, print the percentage for columns less than threshold. Defaults to 100.
    """
    print('\n*****INICIO IMPRIMIR PERCENTAGE UNIQUE VALUES FOR EACH COLUMN******')
    for i in data_frame:
        num = len(np.unique(data_frame[i].values))
        percentage = float(num) / data_frame.shape[0] * 100
        if percentage < threshold:
            print('Column: {} - {} - {:.10f}%'.format(i, num, percentage))
    # print(data_frame.nunique() / len(data_frame))
    print('*****FIM IMPRIMIR PERCENTAGE UNIQUE VALUES FOR EACH COLUMN******')


def simulate_delete_columns_with_low_variance(data_frame: pd.DataFrame, thresholds: np.arange, separate_numeric_columns: bool = False) -> None:
    """Plot and print the simulation of delete columns with low variance. This function works only numeric columns.

    Args:
        data_frame (pd.DataFrame): Data frame to be treated.
        thresholds (np.arange): Thresholds to remove the columns.
        separate_numeric_columns (bool, optional): Separate the numeric columns. Defaults to False.
    """
    print('\n*****INICIO IMPRIMIR SIMULATE DELETE COLUMNS WITH LOW VARIANCE******')
    x, _ = pattern_extraction.create_x_y_data(data_frame=data_frame)
    print('Shape do X antes: {}.'.format(x.shape))

    if separate_numeric_columns:
        x, x_numeric = utils.separate_numeric_columns(x=x)
        x, results = __execute_delete_columns_with_low_variance(
            x=x_numeric, thresholds=thresholds)
        x = np.concatenate((x, x_numeric), axis=1)
    else:
        x, results = __execute_delete_columns_with_low_variance(
            x=x, thresholds=thresholds)

    print('\nShape do X depois: {}.'.format(x.shape))

    plt.plot(thresholds, results)
    plt.show()
    print('\n*****FIM IMPRIMIR SIMULATE DELETE COLUMNS WITH LOW VARIANCE******')

################################################## PRIVATE METHODS ##################################################


def __magnify() -> list:
    """Style a table from a dataframe.

    Returns:
        list: Dataframe table style attributes.
    """

    return [dict(selector="th", props=[("font-size", "7pt")]),
            dict(selector="td", props=[('padding', "0em 0em")]),
            dict(selector="th:hover", props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover", props=[('max-width', '200px'), ('font-size', '12pt')])]


def __generate_random_rgb_color() -> list:
    """Generate a random rgb color.

    Returns:
        list: Random rgb color.
    """
    # return '#%02x%02x%02x' % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 70))


def __execute_delete_columns_with_low_variance(x: np.array, thresholds: np.arange) -> list:
    """Execute the delete columns with low variance.

    Args:
        x (np.array): Numpy array to be treated.
        thresholds (np.arange): Thresholds to remove the columns.

    Returns:
        tuple: Tuple with the x, with features removed, and the results of execution.
    """
    results = list()
    for threshold in thresholds:
        x = pre_processing.delete_columns_with_low_variance(
            x=x, threshold=threshold, separate_numeric_columns=False)
        results.append(x.shape[1])
    return x, results
