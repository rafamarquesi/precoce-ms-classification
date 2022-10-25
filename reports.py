import random
import utils
import pre_processing

from sys import displayhook

import pandas as pd
import numpy as np

import seaborn as sns
import dataframe_image as dfi
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.inspection import permutation_importance


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
    displayhook(df_nan)
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
    pd.set_option('display.max_rows', 10)
    print('Número de atributos: {}'.format(len(data_frame.columns)))
    print('Nome dos atributos: {}'.format(list(data_frame.columns)))
    for column_name, column_data in data_frame.iteritems():
        print('-> Atributo: {}'.format(column_name))
        print('Contagem de valor:\n{}'.format(column_data.value_counts()))
        print('Descrição:\n{}'.format(column_data.describe()))
        print('Número de nan: {}'.format(column_data.isna().sum()))
        print('-------------------------------')
    pd.set_option('display.max_rows', utils.PANDAS_MAX_ROWS)
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


def correlation_matrix(data_frame: pd.DataFrame, method: str, attribute: str = None, display_matrix: bool = False, export_matrix: bool = False, path_save_matrix: str = None, print_corr_matrix_summarized: bool = False, lower_limit: float = -0.5, upper_limit: float = 0.5) -> None:
    """Create a correlation matrix from the DataFrame.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        method (str): Method to be used to create the correlation matrix.
        attribute (str, optional): Attribute to be used to create the correlation matrix. Defaults to None.
        display_matrix (bool, optional): Flag to display the correlation matrix. Defaults to False.
        export_matrix (bool, optional): Flag to export the correlation matrix. Defaults to False.
        path_save_matrix (str, optional): Path to save the correlation matrix. Defaults to None.
        print_corr_matrix_summarized (bool, optional): Flag to display the results of the correlation matrix in a summarized form, that is, the values that are in the range passed as a parameter. Works only when attribute is None. If false, results will not be displayed. Defaults to False.
        lower_limit (float, optional): Lower limit of the interval in the correlation matrix summarized. Defaults to -0.5.
        upper_limit (float, optional): Upper limit of the interval in the correlation matrix summarized. Defaults to 0.5.
    """
    print('\n*****INICIO CORRELATION MATRIX******')
    if attribute is None:
        correlation_matrix = data_frame.corr(method=method).astype('float32')

        if print_corr_matrix_summarized:
            __print_correlation_matrix_summarized(
                correlation_matrix=correlation_matrix, lower_limit=lower_limit, upper_limit=upper_limit)

        cmap = sns.diverging_palette(5, 250, as_cmap=True)

        styled_table = correlation_matrix.style.background_gradient(cmap, axis=1)\
            .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
            .set_caption("Hover to magify")\
            .set_precision(2)\
            .set_table_styles(__magnify())
    else:
        correlation_matrix = data_frame.corr(
            method=method).astype('float32')[attribute]

        correlation_matrix = correlation_matrix.sort_values(ascending=False)

        styled_table = correlation_matrix.to_frame().style.background_gradient(
            cmap=sns.light_palette((260, 75, 60), input="husl", as_cmap=True))

    if display_matrix:
        displayhook(styled_table)

    if export_matrix:
        path_save_matrix = __define_path_save_fig(
            path_save_fig=path_save_matrix)

        name_figure = 'correlation_matrix-{}-{}-{}.png'.format(
            method,
            attribute,
            utils.get_current_datetime()
        )

        dfi.export(
            styled_table,
            ''.join([path_save_matrix, name_figure]),
            max_cols=-1,
            max_rows=-1
        )
        print('Figure {} saved in {} directory.'.format(
            name_figure, path_save_matrix))

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

    path_save_fig = __define_path_save_fig(path_save_fig=path_save_fig)

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
            'evaluation-measures-{}-{}.png'.format(
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

        fig = plt.figure(figsize=(10, 8))
        accuracy_graphic[key].to_frame().boxplot()
        fig.savefig(
            path_save_fig + 'boxplot-accuracy-{}-{}.png'.format(key, utils.get_current_datetime()))

        if display_results:
            plt.show()

    fig = plt.figure(figsize=(10, 8))
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
    displayhook(data_frame.nunique())
    print('*****FIM IMPRIMIR UNIQUE VALUES FOR EACH COLUMN******')


def percentage_unique_values_for_each_column(data_frame: pd.DataFrame, threshold: float = 100) -> None:
    """To help highlight columns of near-zero variance, can calculate the number of unique values for each variable as a percentage of the total number of rows in the dataset.

    Args:
        data_frame (pd.DataFrame): Data frame to be treated.
        threshold (float, optional): If 100, print percentage for all columns. Other threshold, print the percentage for columns less than threshold. Defaults to 100.
    """
    print('\n*****INICIO IMPRIMIR PERCENTAGE UNIQUE VALUES FOR EACH COLUMN******')
    df_percentage = pd.DataFrame()
    for column in data_frame:
        total_unique = len(np.unique(data_frame[column].values))
        percentage = float(total_unique) / data_frame.shape[0] * 100
        if percentage < threshold:
            df_percentage = pd.concat([df_percentage, pd.DataFrame.from_records(
                [
                    {
                        'Coluna': column,
                        'total_Unique': total_unique,
                        'Porcentagem': percentage
                    }
                ])])
    df_percentage = df_percentage.sort_values(
        by=['Porcentagem'], ascending=[True])
    displayhook(df_percentage)
    print('*****FIM IMPRIMIR PERCENTAGE UNIQUE VALUES FOR EACH COLUMN******')


def simulate_delete_columns_with_low_variance(data_frame: pd.DataFrame, thresholds: np.arange, separate_numeric_columns: bool = False, path_save_fig: str = None, display_figure: bool = False) -> None:
    """Plot and print the simulation of delete columns with low variance. This function works only numeric columns.

    Args:
        data_frame (pd.DataFrame): Data frame to be treated.
        thresholds (np.arange): Thresholds to remove the columns.
        separate_numeric_columns (bool, optional): Separate the numeric columns. Defaults to False.
        path_save_fig (str, optional): Path to save the figures. If None, save the figures in root path of project. Defaults to None.
        display_figure (bool, optional): Flag to display the results, for example in jupyter notebook. Defaults to False.

    """
    print('\n*****INICIO IMPRIMIR SIMULATE DELETE COLUMNS WITH LOW VARIANCE******')
    x, _ = utils.create_x_y_dataframe_data(data_frame=data_frame)
    print('Shape do X antes: {}.'.format(x.shape))

    if separate_numeric_columns:
        x, x_numeric = utils.separate_numeric_dataframe_columns(
            x=x,
            exclude_types=utils.TYPES_EXCLUDE_DF
        )
        x_numeric, results = __execute_delete_columns_with_low_variance(
            x=x_numeric, thresholds=thresholds)
        x = utils.concatenate_data_frames(data_frames=[x, x_numeric])
    else:
        x, results = __execute_delete_columns_with_low_variance(
            x=x, thresholds=thresholds)

    print('\nShape do X depois: {}.'.format(x.shape))

    path_save_fig = __define_path_save_fig(path_save_fig=path_save_fig)

    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, results)
    name_figure = 'plot-simulate-delete-columns-with-low-variance-{}.png'.format(
        utils.get_current_datetime())
    plt.savefig(''.join([path_save_fig, name_figure]))
    print('Figure {} saved in {} directory.'.format(name_figure, path_save_fig))

    if display_figure:
        plt.show()

    plt.close()

    print('\n*****FIM IMPRIMIR SIMULATE DELETE COLUMNS WITH LOW VARIANCE******')


def feature_importance_using_coefficients_of_linear_models(data_frame: pd.DataFrame, models: list, path_save_fig: str = None, display_figure: bool = False) -> None:
    """Print the feature importance using coefficients of linear models.
    The models supported are: LogisticRegression (logistic_regression), LinearSVC (linear_svc), SGDClassifier (sgd_classifier).

    Args:
        data_frame (pd.DataFrame): Data frame to be treated.
        models (list): Models to be used.
        path_save_fig (str, optional): Path to save the figures. If None, save the figures in root path of project. Defaults to None.
        display_figure (bool, optional): Flag to display the results, for example in jupyter notebook. Defaults to False.
    """
    print('\n*****INICIO IMPRIMIR FEATURE IMPORTANCE USING COEFFICIENTS OF LINEAR MODELS******')
    if models:
        path_save_fig = __define_path_save_fig(path_save_fig=path_save_fig)

        x, y = utils.create_x_y_dataframe_data(data_frame=data_frame)

        for model in models:
            if model == 'logistic_regression':
                model = LogisticRegression()
            elif model == 'linear_svc':
                model = LinearSVC()
            elif model == 'sgd_classifier':
                model = SGDClassifier()
            else:
                raise Exception('Model not supported.')

            model.fit(x, y)
            print('\n\nModel: {}'.format(model))
            print('\nFeature importance using coefficients of linear models:')
            importance = model.coef_[0]
            displayhook(pd.DataFrame(
                {
                    'Feature': x.columns,
                    'Importance': importance
                }
            ).sort_values(by=['Importance'], ascending=[False]))

            plt.figure(figsize=(10, 8))
            plt.bar([x for x in range(len(importance))], importance)
            plt.title('Model: {}'.format(model.__class__.__name__))
            name_figure = 'bar-feature_importance_using_coefficients_of_linear_models-{}-{}.png'.format(
                model.__class__.__name__,
                utils.get_current_datetime()
            )
            plt.savefig(''.join([path_save_fig, name_figure]))
            print('Figure {} saved in {} directory.'.format(
                name_figure, path_save_fig))

            if display_figure:
                plt.show()

            plt.close()
    else:
        raise Exception('Models not informed.')

    print('*****FIM IMPRIMIR FEATURE IMPORTANCE USING COEFFICIENTS OF LINEAR MODELS******')


def feature_importance_using_tree_based_models(data_frame: pd.DataFrame, models: list, path_save_fig: str = None, display_figure: bool = False) -> None:
    """Print the feature importance using tree based models.
    The models supported are: DecisionTreeClassifier (decision_tree_classifier), RandomForestClassifier (random_forest_classifier), XGBClassifier (xgb_classifier).
    Args:
        data_frame (pd.DataFrame): Data frame to be treated.
        models (list): Models to be used.
        path_save_fig (str, optional): Path to save the figures. If None, save the figures in root path of project. Defaults to None.
        display_figure (bool, optional): Flag to display the results, for example in jupyter notebook. Defaults to False.
    """
    print('\n*****INICIO IMPRIMIR FEATURE IMPORTANCE USING TREE BASED MODELS******')
    if models:
        path_save_fig = __define_path_save_fig(path_save_fig=path_save_fig)

        x, y = utils.create_x_y_dataframe_data(data_frame=data_frame)

        for model in models:
            if model == 'decision_tree_classifier':
                model = DecisionTreeClassifier()
            elif model == 'random_forest_classifier':
                model = RandomForestClassifier()
            elif model == 'xgb_classifier':
                model = XGBClassifier()
            else:
                raise Exception('Model not supported.')

            model.fit(x, y)
            print('\n\nModel: {}'.format(model))
            print('\nFeature importance using tree based models:')
            importance = model.feature_importances_
            displayhook(pd.DataFrame(
                {
                    'Feature': x.columns,
                    'Importance': importance
                }
            ).sort_values(by=['Importance'], ascending=[False]))

            plt.figure(figsize=(10, 8))
            plt.bar([x for x in range(len(importance))], importance)
            plt.title('Model: {}'.format(model.__class__.__name__))
            name_figure = 'bar-feature_importance_using_tree_based_models-{}-{}.png'.format(
                model.__class__.__name__,
                utils.get_current_datetime()
            )
            plt.savefig(''.join([path_save_fig, name_figure]))
            print('Figure {} saved in {} directory.'.format(
                name_figure, path_save_fig))

            if display_figure:
                plt.show()

            plt.close()
    else:
        raise Exception('Models not informed.')

    print('*****FIM IMPRIMIR FEATURE IMPORTANCE USING TREE BASED MODELS******')


def feature_importance_using_permutation_importance(data_frame: pd.DataFrame, models: list, path_save_fig: str = None, display_figure: bool = False) -> None:
    """Print the feature importance using permutation importance.
    The models supported are: KNeighborsClassifier (knneighbors_classifier), GaussianNB (gaussian_nb).

    Args:
        data_frame (pd.DataFrame): Data frame to be treated.
        models (list): Models to be used.
        path_save_fig (str, optional): Path to save the figures. If None, save the figures in root path of project. Defaults to None.
        display_figure (bool, optional): Flag to display the results, for example in jupyter notebook. Defaults to False.
    """
    print('\n*****INICIO IMPRIMIR FEATURE IMPORTANCE USING PERMUTATION IMPORTANCE******')
    if models:
        path_save_fig = __define_path_save_fig(path_save_fig=path_save_fig)

        x, y = utils.create_x_y_dataframe_data(data_frame=data_frame)

        for model in models:
            if model == 'knneighbors_classifier':
                model = KNeighborsClassifier()
            elif model == 'gaussian_nb':
                model = GaussianNB()
            else:
                raise Exception('Model not supported.')

            model.fit(x, y)
            results = permutation_importance(
                model, x, y, scoring='accuracy', n_repeats=5, random_state=42, n_jobs=-1)
            print('\n\nModel: {}'.format(model))
            print('\nFeature importance using tree based models:')
            importance = results.importances_mean
            displayhook(pd.DataFrame(
                {
                    'Feature': x.columns,
                    'Importance': importance
                }
            ).sort_values(by=['Importance'], ascending=[False]))

            plt.figure(figsize=(10, 8))
            plt.bar([x for x in range(len(importance))], importance)
            plt.title('Model: {}'.format(model.__class__.__name__))
            name_figure = 'bar-feature_importance_using_tree_based_models-{}-{}.png'.format(
                model.__class__.__name__,
                utils.get_current_datetime()
            )
            plt.savefig(''.join([path_save_fig, name_figure]))
            print('Figure {} saved in {} directory.'.format(
                name_figure, path_save_fig))

            if display_figure:
                plt.show()

            plt.close()
    else:
        raise Exception('Models not informed.')

    # sorted_idx = result.importances_mean.argsort()

    # fig, ax = plt.subplots()
    # ax.boxplot(result.importances[sorted_idx].T,
    #            vert=False, labels=x.columns[sorted_idx])
    # ax.set_title("Permutation Importances (test set)")
    # fig.tight_layout()
    # plt.show()

    print('*****FIM IMPRIMIR FEATURE IMPORTANCE USING PERMUTATION IMPORTANCE******')


def histogram(data_frame: pd.DataFrame, save_fig: bool = False, path_save_fig: str = None, display_figure: bool = True) -> None:
    """Print the histogram for each attribute of the data frame.
    Args:
        data_frame (pd.DataFrame): Data frame to be treated.
        save_fig (bool, optional): Flag to save the figures. Defaults to False.
        path_save_fig (str, optional): Path to save the figures. If None, save the figures in root path of project. Defaults to None.
        display_figure (bool, optional): Flag to display the results, for example in jupyter notebook. Defaults to True.
    """
    print('\n*****INICIO IMPRIMIR HISTOGRAM******')
    for column in data_frame.columns:
        plt.figure(figsize=(10, 8))
        plt.hist(data_frame[column])
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        plt.title('Histogram of {}'.format(column))
        if save_fig:
            path_save_fig = __define_path_save_fig(path_save_fig=path_save_fig)

            name_figure = 'histogram-{}-{}.png'.format(
                column, utils.get_current_datetime())
            plt.savefig(
                ''.join([path_save_fig, name_figure]), bbox_inches='tight')
            print('Figure {} saved in {} directory.'.format(
                name_figure, path_save_fig))

        if display_figure:
            plt.show()

        plt.close()
        print('\n')
    print('*****FIM IMPRIMIR HISTOGRAM******')


def histogram_grouped_by_target(data_frame: pd.DataFrame, target: str, save_fig: bool = False, path_save_fig: str = None, display_figure: bool = True) -> None:
    """Print the histogram for each attribute of the data frame grouped by target.
    Args:
        data_frame (pd.DataFrame): Data frame to be treated.
        target (str): Target to be grouped.
        save_fig (bool, optional): Flag to save the figures. Defaults to False.
        path_save_fig (str, optional): Path to save the figures. If None, save the figures in root path of project. Defaults to None.
        display_figure (bool, optional): Flag to display the results, for example in jupyter notebook. Defaults to True.
    """
    print('\n*****INICIO IMPRIMIR HISTOGRAM GROUPED BY TARGET******')

    try:
        target_attr_unique = data_frame[target].unique()
    except:
        raise Exception('Target ("{}") not found in DataFrame.'.format(target))

    colors = [
        'red', 'green', 'blue', 'yellow', 'black', 'orange',
        'purple', 'pink', 'brown', 'gray', 'olive', 'cyan'
    ]

    kwargs = dict(alpha=0.5, bins=10)

    for column in data_frame.columns:
        plt.figure(figsize=(10, 8))
        for i in range(len(target_attr_unique)):
            x = data_frame.loc[
                data_frame[target] == target_attr_unique[i],
                column
            ]

            plt.hist(x, **kwargs, color=colors[i], label=target_attr_unique[i])

        plt.gca().set(title='Histogram of {} grouped by {}'.format(
            column, target), ylabel='Frequency')
        plt.xticks(rotation=80, fontsize=7)
        plt.tight_layout()
        plt.legend()
        if save_fig:
            path_save_fig = __define_path_save_fig(
                path_save_fig=path_save_fig)

            name_figure = 'histogram_grouped_by_target-{}-{}-{}.png'.format(
                column, target, utils.get_current_datetime())
            plt.savefig(
                ''.join([path_save_fig, name_figure]), bbox_inches='tight')
            print('Figure {} saved in {} directory.'.format(
                name_figure, path_save_fig))

        if display_figure:
            plt.show()

        plt.close()
        print('\n')
    print('*****FIM IMPRIMIR HISTOGRAM GROUPED BY TARGET******')


def boxplot(data_frame: pd.DataFrame, save_fig: bool = False, path_save_fig: str = None, display_figure: bool = True) -> None:
    """Print the boxplot for each attribute of the data frame.
    Args:
        data_frame (pd.DataFrame): Data frame to be treated.
        save_fig (bool, optional): Flag to save the figures. Defaults to False.
        path_save_fig (str, optional): Path to save the figures. If None, save the figures in root path of project. Defaults to None.
        display_figure (bool, optional): Flag to display the results, for example in jupyter notebook. Defaults to True.
    """
    print('\n*****INICIO IMPRIMIR BOXPLOT******')
    for column in data_frame.select_dtypes(exclude=utils.TYPES_EXCLUDE_DF).columns:
        plt.figure(figsize=(5, 10))
        plt.boxplot(data_frame[column], labels=[column])
        plt.xticks(rotation='horizontal')
        plt.tight_layout()
        plt.title('Boxplot of {}'.format(column))
        if save_fig:
            path_save_fig = __define_path_save_fig(
                path_save_fig=path_save_fig)

            name_figure = 'boxplot-{}-{}.png'.format(
                column, utils.get_current_datetime())
            plt.savefig(
                ''.join([path_save_fig, name_figure]), bbox_inches='tight')
            print('Figure {} saved in {} directory.'.format(
                name_figure, path_save_fig))

        if display_figure:
            plt.show()

        plt.close()
        print('\n')
    print('*****FIM IMPRIMIR BOXPLOT******')


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


def __execute_delete_columns_with_low_variance(x: pd.DataFrame, thresholds: np.arange) -> list:
    """Execute the delete columns with low variance.

    Args:
        x (pd.DataFrame): Pandas DataFrame with numeric data to be treated.
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


def __define_path_save_fig(path_save_fig: str) -> str:
    """Define the path to save the figures.

    Args:
        path_save_fig (str): Path to save the figures.

    Returns:
        str: Path to save the figures.
    """
    if path_save_fig is None:
        path_save_fig = ''
    else:
        path_save_fig = path_save_fig + '/'
    return path_save_fig


def __print_correlation_matrix_summarized(correlation_matrix: pd.DataFrame, lower_limit: float, upper_limit: float) -> None:
    """Print the correlation matrix summarized.

    Args:
        correlation_matrix (pd.DataFrame): Correlation matrix.
        lower_limit (float): Lower limit of the interval.
        upper_limit (float): Upper limit of the interval.
    """
    indexes = correlation_matrix.index
    columns = correlation_matrix.columns

    correlation_summarized = pd.DataFrame(columns=['Corr_Between', 'Value'])

    keyword = '-and-'
    for column in columns:
        for index in indexes:
            value = correlation_matrix.loc[index, column]
            if ((value < lower_limit) or (value > upper_limit)):
                if ((keyword.join([index, column]) not in correlation_summarized['Corr_Between'].values) and (keyword.join([column, index]) not in correlation_summarized['Corr_Between'].values)):
                    correlation_summarized = pd.concat([correlation_summarized, pd.DataFrame.from_records(
                        [
                            {
                                'Corr_Between': keyword.join([index, column]),
                                'Value': value
                            }
                        ])])

    correlation_summarized = correlation_summarized.sort_values(
        by=['Value'], ascending=False)

    print('Correlation summarized:')
    displayhook(correlation_summarized)
