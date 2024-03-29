import random
from typing import Union

import utils
import pre_processing
import settings

from sys import displayhook

import pandas as pd
import numpy as np

import seaborn as sns
import dataframe_image as dfi
import matplotlib.pyplot as plt
import missingno as msno

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV

from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


@utils.timeit
def informations(data_frame: pd.DataFrame) -> None:
    """Print some informations of the DataFrame.

    Args:
        data_frame (pd.DataFrame): DataFrame to be printed.
    """

    print('\n*****INICIO PRINT INFOS******')
    print('Número total de linhas do DataFrame: {}'.format(len(data_frame.index)))
    print('Número de colunas: {}'.format(data_frame.columns.size))
    print('Informações do DataFrame:')
    data_frame.info(verbose=True, memory_usage='deep')
    print('*****FIM PRINT INFOS*********')


@utils.timeit
def print_list_columns(data_frame: pd.DataFrame) -> None:
    """Show the columns of the DataFrame.

    Args:
        data_frame (pd.DataFrame): DataFrame to be printed.
    """
    print('\n*****INICIO SHOW COLUNAS******')
    print(list(data_frame.columns))
    print('*****FIM SHOW COLUNAS*********')


@utils.timeit
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
    print('A matrix representation of the missing values (white) by variable')
    msno.matrix(data_frame)
    print('*****FIM RELATÓRIO ATRIBUTOS NAN******')


@utils.timeit
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
                column_data.shift().bfill().astype(column_data.dtype.name)).values
            if not np.all(comparison == comparison[0]):
                print('Nome coluna que diverge: {}'.format(column_name))
                print('Index das linhas e valor na coluna que diverge:\n{}'.format(
                    column_data))
                # print(column_data.ne(column_data.shift().bfill()).astype('uint8'))
                print('-------------------------------')
        print('Próximo ++++++++++++++')
    print('*****FIM RELATÓRIO LINHAS DUPLICADAS******')


@utils.timeit
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
    pd.set_option('display.max_rows', settings.PANDAS_MAX_ROWS)
    print('*****FIM RELATÓRIO ATRIBUTOS******')


@utils.timeit
def class_distribution(y: np.array) -> None:
    """Report of class distribution.

    Args:
        y (np.array): NumPy Array to be treated.
    """
    print('\n*****INICIO RELATÓRIO DISTRIBUIÇÃO DE CLASSES******')
    unique, counts = np.unique(y, return_counts=True)
    if unique.size == 2:
        dist_1 = y[y.nonzero()].size / y.size
        dist_0 = (y.size - y[y.nonzero()].size) / y.size
        print('Distribuição da classe 1: {0:.0%}'.format(dist_1))
        print('Distribuição da classe 0: {0:.0%}'.format(dist_0))
        if dist_1 > dist_0:
            print('Erro majoritário: {0:.0%}'.format(1 - dist_1))
        else:
            print('Erro majoritário: {0:.0%}'.format(1 - dist_0))
    else:
        for key, value in dict(zip(unique, counts)).items():
            print('Distribuição da classe {}: {:.0%}'.format(key, value/y.size))
    print('*****FIM RELATÓRIO DISTRIBUIÇÃO DE CLASSES******')


@utils.timeit
def correlation_matrix(data_frame: pd.DataFrame, method: str, attribute: str = None, display_matrix: bool = False, export_matrix: bool = False, path_save_matrix: str = None, print_corr_matrix_summarized: bool = False, lower_limit: float = -0.5, upper_limit: float = 0.5, to_latex: bool = False) -> None:
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
        to_latex (bool, optional): Flag to export the correlation matrix to latex. Defaults to False.
    """
    print('\n*****INICIO CORRELATION MATRIX******')
    if attribute is None:
        correlation_matrix = data_frame.corr(method=method).astype('float32')

        if print_corr_matrix_summarized:
            __print_correlation_matrix_summarized(
                correlation_matrix=correlation_matrix, lower_limit=lower_limit, upper_limit=upper_limit, to_latex=to_latex)

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

        if to_latex:
            print('Correlantion matrix in latex format:')
            print(correlation_matrix.to_latex(
                index=False, float_format="%.2f"))

        styled_table = correlation_matrix.to_frame().style.background_gradient(
            cmap=sns.light_palette((260, 75, 60), input="husl", as_cmap=True))

    if display_matrix:
        displayhook(styled_table)

    if export_matrix:
        path_save_matrix = utils.define_path_save_file(
            path_save_file=path_save_matrix)

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


@utils.timeit
def models_results(models_results: dict, path_save_fig: str = None, display_results: bool = False) -> None:
    """Print the models results.

    Args:
        models_results (dict): Results of the models.
        path_save_fig (str, optional): Path to save the figures. If None, save the figures in root path of project. Defaults to None.
        display_results (bool, optional): Flag to display the results, for example in jupyter notebook. Defaults to False.
    """
    print('\n*****INICIO IMPRIMIR RESULTADOS MODELOS******')
    accuracy_graphic = pd.DataFrame()

    path_save_fig = utils.define_path_save_file(path_save_file=path_save_fig)

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


@utils.timeit
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


@utils.timeit
def unique_values_for_each_column(data_frame: pd.DataFrame) -> None:
    """Print the unique values for each column in the data frame.

    Args:
        data_frame (pd.DataFrame): Data frame to be treated.
    """
    print('\n*****INICIO IMPRIMIR UNIQUE VALUES FOR EACH COLUMN******')
    displayhook(data_frame.nunique())
    print('*****FIM IMPRIMIR UNIQUE VALUES FOR EACH COLUMN******')


@utils.timeit
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


@utils.timeit
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

    path_save_fig = utils.define_path_save_file(path_save_file=path_save_fig)

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


@utils.timeit
def feature_importance_using_coefficients_of_linear_models(data_frame: pd.DataFrame, models: list, path_save_fig: str = None, display_figure: bool = False, class_weight: Union[dict, str, None] = None, n_jobs: int = -1) -> None:
    """Print the feature importance using coefficients of linear models.
    The models supported are: LogisticRegression (logistic_regression), LinearSVC (linear_svc), SGDClassifier (sgd_classifier).

    Args:
        data_frame (pd.DataFrame): Data frame to be treated.
        models (list): Models to be used.
        path_save_fig (str, optional): Path to save the figures. If None, save the figures in root path of project. Defaults to None.
        display_figure (bool, optional): Flag to display the results, for example in jupyter notebook. Defaults to False.
        class_weight (Union[dict, str, None], optional): Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y. Defaults to None.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.
    """
    print('\n*****INICIO IMPRIMIR FEATURE IMPORTANCE USING COEFFICIENTS OF LINEAR MODELS******')
    if models:
        path_save_fig = utils.define_path_save_file(
            path_save_file=path_save_fig)

        x, y = utils.create_x_y_dataframe_data(data_frame=data_frame)

        for model in models:
            if model == 'logistic_regression':
                model = LogisticRegression(
                    class_weight=class_weight, n_jobs=n_jobs)
            elif model == 'linear_svc':
                model = LinearSVC(class_weight=class_weight)
            elif model == 'sgd_classifier':
                model = SGDClassifier(n_jobs=n_jobs, class_weight=class_weight)
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


@utils.timeit
def feature_importance_using_tree_based_models(data_frame: pd.DataFrame, models: list, path_save_fig: str = None, display_figure: bool = False, class_weight: Union[dict, str, None] = None, n_jobs: int = -1) -> None:
    """Print the feature importance using tree based models.
    The models supported are: DecisionTreeClassifier (decision_tree_classifier), RandomForestClassifier (random_forest_classifier), XGBClassifier (xgb_classifier).
    Args:
        data_frame (pd.DataFrame): Data frame to be treated.
        models (list): Models to be used.
        path_save_fig (str, optional): Path to save the figures. If None, save the figures in root path of project. Defaults to None.
        display_figure (bool, optional): Flag to display the results, for example in jupyter notebook. Defaults to False.
        class_weight (Union[dict, str, None], optional): Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y. Defaults to None.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.
    """
    print('\n*****INICIO IMPRIMIR FEATURE IMPORTANCE USING TREE BASED MODELS******')
    if models:
        path_save_fig = utils.define_path_save_file(
            path_save_file=path_save_fig)

        x, y = utils.create_x_y_dataframe_data(data_frame=data_frame)

        for model in models:
            if model == 'decision_tree_classifier':
                model = DecisionTreeClassifier(class_weight=class_weight)
            elif model == 'random_forest_classifier':
                model = RandomForestClassifier(
                    n_jobs=n_jobs, class_weight=class_weight)
            elif model == 'xgb_classifier':
                model = XGBClassifier(n_jobs=n_jobs)
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


@utils.timeit
def feature_importance_using_permutation_importance(data_frame: pd.DataFrame, models: list, scoring: str = 'accuracy', n_repeats: int = 5, random_state: int = 42, n_jobs: int = -1, path_save_fig: str = None, display_figure: bool = False) -> None:
    """Print the feature importance using permutation importance.
    The models supported are: KNeighborsClassifier (knneighbors_classifier), GaussianNB (gaussian_nb).

    Args:
        data_frame (pd.DataFrame): Data frame to be treated.
        models (list): Models to be used. The models supported are: KNeighborsClassifier (knneighbors_classifier), GaussianNB (gaussian_nb).
        scoring (str, optional): Scoring to be used. Defaults to 'accuracy'.
        n_repeats (int, optional): Number of times to permute a feature. Defaults to 5.
        random_state (int, optional): Random state. Defaults to 42.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.
        path_save_fig (str, optional): Path to save the figures. If None, save the figures in root path of project. Defaults to None.
        display_figure (bool, optional): Flag to display the results, for example in jupyter notebook. Defaults to False.
    """
    print('\n*****INICIO IMPRIMIR FEATURE IMPORTANCE USING PERMUTATION IMPORTANCE******')
    if models:
        print('!!>>Scoring: {}'.format(scoring))

        path_save_fig = utils.define_path_save_file(
            path_save_file=path_save_fig)

        x, y = utils.create_x_y_dataframe_data(data_frame=data_frame)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, random_state=random_state)

        for model in models:
            if model == 'knneighbors_classifier':
                model = KNeighborsClassifier(n_jobs=n_jobs)
            elif model == 'gaussian_nb':
                model = GaussianNB()
            else:
                raise Exception('Model not supported.')

            model.fit(x_train, y_train)
            print('\n\nModel {} fited.'.format(model.__class__.__name__))
            print('Model {} score: {}'.format(
                model.__class__.__name__, model.score(x_test, y_test)))
            print('Appling permutation importance...')

            results = permutation_importance(
                model, x_test, y_test, scoring=scoring, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs)

            for i in results.importances_mean.argsort()[::-1]:
                if results.importances_mean[i] - 2 * results.importances_std[i] > 0:
                    print('Feature: {} - Mean: {:.3f} - Std: +/- {:.3f}'.format(
                        x.columns[i], results.importances_mean[i], results.importances_std[i]))

            print('Model: {}'.format(model))
            print('\nPermutation importance:')
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
            name_figure = 'bar-feature_importance_using_permutation_importance-{}-{}.png'.format(
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


@utils.timeit
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
        plt.xticks(rotation=80, fontsize=7)
        plt.tight_layout()
        plt.title('Histograma de {}'.format(column))
        if save_fig:
            path_save_fig = utils.define_path_save_file(
                path_save_file=path_save_fig)

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


@utils.timeit
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

        plt.gca().set(title='Histograma de {} agrupado por {}'.format(
            column, target), ylabel='Frequency')
        plt.xticks(rotation=80, fontsize=7)
        plt.tight_layout()
        plt.legend()
        if save_fig:
            path_save_fig = utils.define_path_save_file(
                path_save_file=path_save_fig)

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


@utils.timeit
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
        plt.title('Boxplot de {}'.format(column))
        if save_fig:
            path_save_fig = utils.define_path_save_file(
                path_save_file=path_save_fig)

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


@utils.timeit
def boxplot_grouped_by_target(data_frame: pd.DataFrame, target: str, save_fig: bool = False, path_save_fig: str = None, display_figure: bool = True) -> None:
    """Print the boxplot for each attribute of the data frame grouped by target.
    Args:
        data_frame (pd.DataFrame): Data frame to be treated.
        target (str): Target to be grouped.
        save_fig (bool, optional): Flag to save the figures. Defaults to False.
        path_save_fig (str, optional): Path to save the figures. If None, save the figures in root path of project. Defaults to None.
        display_figure (bool, optional): Flag to display the results, for example in jupyter notebook. Defaults to True.
    """
    print('\n*****INICIO IMPRIMIR BOXPLOT GROUPED BY TARGET******')

    try:
        target_attr_unique = data_frame[target].unique()
    except:
        raise Exception('Target ("{}") not found in DataFrame.'.format(target))

    for column in data_frame.select_dtypes(exclude=utils.TYPES_EXCLUDE_DF).columns:
        plt.figure(figsize=(5, 10))

        x = list()
        for i in range(len(target_attr_unique)):
            x.append(data_frame.loc[
                data_frame[target] == target_attr_unique[i],
                column
            ])

        plt.boxplot(x, labels=target_attr_unique)

        plt.xticks(rotation='horizontal')
        plt.tight_layout()
        plt.title('Boxplot de {} agrupado por {}'.format(column, target))
        if save_fig:
            path_save_fig = utils.define_path_save_file(
                path_save_file=path_save_fig)

            name_figure = 'boxplot_grouped_by_target-{}-{}-{}.png'.format(
                column, target, utils.get_current_datetime())
            plt.savefig(
                ''.join([path_save_fig, name_figure]), bbox_inches='tight')
            print('Figure {} saved in {} directory.'.format(
                name_figure, path_save_fig))

        if display_figure:
            plt.show()

        plt.close()
        print('\n')
    print('*****FIM IMPRIMIR BOXPLOT GROUPED BY TARGET******')


@utils.timeit
def detect_outiliers_from_attribute(data_frame: pd.DataFrame, attribute_name: str) -> None:
    """Detect outiliers from attribute, and print the outiliers.
    Args:
        data_frame (pd.DataFrame): Data frame to be treated.
        attribute_name (str): Attribute to be treated.
    """
    print('\n*****INICIO DETECT OUTILIERS FROM ATTRIBUTE******')

    if not attribute_name in data_frame.columns:
        raise Exception(
            'Attribute ("{}") not found in DataFrame.'.format(attribute_name))

    print('Outiliers for attribute "{}":'.format(attribute_name))

    displayhook(
        data_frame[
            pre_processing.detect_outliers(
                series=data_frame[attribute_name]
            )
        ][attribute_name].sort_values(ascending=False)
    )

    print('\n*****FIM DETECT OUTILIERS FROM ATTRIBUTE******')


@utils.timeit
def simulate_sequential_feature_selector(data_frame: pd.DataFrame, estimator: object, k_features: Union[int, tuple, str], forward: bool = True, floating: bool = False, verbose: int = 0, scoring: str = 'accuracy', cv: int = 0, n_jobs: int = -1, pre_dispatch: str = '2*n_jobs', clone_estimator: bool = True, display_figure: bool = True, save_fig: bool = False, path_save_fig: str = None) -> None:
    """Simulate the sequential feature selector, using the mlxtend library.
    Sequential Feature Selector, from mlxtend, is a wrapper class that allows you to perform forward, backward, or floating sequential selection of features.
    The feature selections available are: Sequential Forward Selection (SFS), Sequential Backward Selection (SBS), Sequential Forward Floating Selection (SFFS), Sequential Backward Floating Selection (SBFS).
    For more information, see: http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/.

    Args:
        data_frame (pd.DataFrame): Data frame to be treated.
        estimator (object): A scikit-learn classifier or regressor.
        k_features (Union[int, tuple, str]): The number of features to select, where k_features < the full feature set. New in version 0.4.2: A tuple containing a min and max value can be provided as well, e.g., (1, 4) to select between 1 and 4 features inclusive. A string argument in {'best', 'parsimonious'} is accepted as well (new in v0.8.0).
        forward (bool, optional): Perform forward selection. Defaults to True.
        floating (bool, optional): Adds a conditional exclusion/inclusion if True. Defaults to False.
        verbose (int, optional): Controls the verbosity of the output. Defaults to 0.
        scoring (str, optional): Scoring metric. Defaults to 'accuracy'.
        cv (int, optional): Number of folds. Defaults to 0.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.
        pre_dispatch (str, optional): Controls the number of jobs that get dispatched during parallel execution. Defaults to '2*n_jobs'.
        clone_estimator (bool, optional): If True, the estimator will be cloned before fitting. Defaults to True.
        display_figure (bool, optional): Flag to display the results, for example in jupyter notebook. Defaults to True.
        save_fig (bool, optional): Flag to save the figures. Defaults to False.
        path_save_fig (str, optional): Path to save the figures. If None, save the figures in root path of project. Defaults to None.
    """
    print('\n*****INICIO SIMULATE SEQUENTIAL FEATURE SELECTOR******')

    x, y = utils.create_x_y_dataframe_data(data_frame=data_frame)

    sfs = SequentialFeatureSelector(
        estimator=estimator,
        k_features=k_features,
        forward=forward,
        floating=floating,
        verbose=verbose,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        pre_dispatch=pre_dispatch,
        clone_estimator=clone_estimator
    )

    sfs = sfs.fit(x, y)

    # print('Selected feature indices at each step:\n{}'.format(sfs.subsets_))

    print('The indices of the best features: {}'.format(sfs.k_feature_idx_))

    print('The feature names of the best features: {}'.format(sfs.k_feature_names_))

    print('The cross-validation score of the best subset: {}'.format(sfs.k_score_))

    print('Output from the feature selection in Data Frame:')
    displayhook(pd.DataFrame.from_dict(sfs.get_metric_dict()).T)

    fig = plot_sfs(sfs.get_metric_dict(), kind='std_dev', figsize=(10, 8))
    plt.title('Sequential Forward Selection (w. StdDev)')
    plt.grid()

    if save_fig:
        path_save_fig = utils.define_path_save_file(
            path_save_file=path_save_fig)

        name_figure = 'sequential_feature_selector-forward_{}-floating_{}-{}.png'.format(
            forward, floating,
            utils.get_current_datetime())
        plt.savefig(
            ''.join([path_save_fig, name_figure]), bbox_inches='tight')
        print('Figure {} saved in {} directory.'.format(
            name_figure, path_save_fig))

    if display_figure:
        plt.show()

    plt.close()

    print('*****FIM SIMULATE SEQUENTIAL FEATURE SELECTOR******')


@utils.timeit
def simulate_recursive_feature_elimination_with_cv(data_frame: pd.DataFrame, estimator: object, step: int = 1, min_features_to_select: int = 1, cv: object = None, scoring: str = None, n_jobs: int = None, verbose: int = 0, display_figure: bool = True, save_fig: bool = False, path_save_fig: str = None) -> None:
    """Simulate the recursive feature elimination with cross-validation, using the sklearn library.
    The Recursive Feature Elimination with cross-validation (RFECV) is a feature selection algorithm that fits a model and removes the weakest feature (or features) until the specified number of features is reached.
    For more information, see: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV.

    Args:
        data_frame (pd.DataFrame): Data frame to be treated.
        estimator (object): A scikit-learn classifier or regressor.
        step (int, optional): The number of features to remove at each iteration. Defaults to 1.
        min_features_to_select (int, optional): The minimum number of features to be selected. Defaults to 1.
        cv (object, optional): Determines the cross-validation splitting strategy. Defaults to None.
        scoring (str, optional): A string (see model evaluation documentation) or a scorer callable object / function with signature scorer(estimator, X, y) which should return only a single value. Defaults to None.
        n_jobs (int, optional): The number of CPUs to use for cross validation. Defaults to None.
        verbose (int, optional): Controls the verbosity of the output. Defaults to 0.
        display_figure (bool, optional): Flag to display the results, for example in jupyter notebook. Defaults to True.
        save_fig (bool, optional): Flag to save the figures. Defaults to False.
        path_save_fig (str, optional): Path to save the figures. If None, save the figures in root path of project. Defaults to None.
    """

    print('*****INICIO SIMULATE RECURSIVE FEATURE ELIMINATION CV******')

    x, y = utils.create_x_y_dataframe_data(data_frame=data_frame)

    rfecv = RFECV(
        estimator=estimator,
        step=step,
        min_features_to_select=min_features_to_select,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose
    )

    rfecv.fit(x, y)

    for i, column in enumerate(x.columns):
        print('Column: {}, Selected {}, Rank: {:.3f}'.format(
            column, rfecv.support_[i], rfecv.ranking_[i]))

    print('Optimal number of features : {}'.format(rfecv.n_features_))

    plt.figure(figsize=(10, 8))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (accuracy)")
    plt.plot(
        range(
            min_features_to_select,
            len(rfecv.cv_results_['mean_test_score']) + min_features_to_select
        ),
        rfecv.cv_results_['mean_test_score'],
    )
    plt.grid()

    if save_fig:
        path_save_fig = utils.define_path_save_file(
            path_save_file=path_save_fig)

        name_figure = 'recursive_feature_elimination_cv-{}.png'.format(
            utils.get_current_datetime())
        plt.savefig(
            ''.join([path_save_fig, name_figure]), bbox_inches='tight')
        print('Figure {} saved in {} directory.'.format(
            name_figure, path_save_fig))

    if display_figure:
        plt.show()

    plt.close()

    print('*****FIM SIMULATE RECURSIVE FEATURE ELIMINATION CV******\n')


def show_settings(settings: object) -> None:
    """Show the settings of the project.

    Args:
        settings (object): Settings of the project.
    """

    print('*****INICIO SHOW SETTINGS******')
    for key, value in settings.__dict__.items():
        if not (key[:2] == '__' and key[-2:] == '__'):
            print('{} = {}'.format(key, value))
    print('*****FIM SHOW SETTINGS******\n')


@utils.timeit
def confusion_matrix_display(y_true: np.array, y_pred: np.array, display_figure: bool = True, save_fig: bool = True, path_save_fig: str = None) -> None:
    """Display the confusion matrix, based in ConfusionMatrixDisplay.from_predictions from scikit-learn, and save it in a file.

    Args:
        y_true (np.array): True labels.
        y_pred (np.array): Predicted labels.
        display_figure (bool, optional): Flag to display the results, for example in jupyter notebook. Defaults to False.
        save_fig (bool, optional): Flag to save the figures. Defaults to False.
        path_save_fig (str, optional): Path to save the figures. If None, save the figures in root path of project. Defaults to None.
    """

    print('*****INICIO CONFUSION MATRIX DISPLAY******')

    plt.figure(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)

    if save_fig:
        path_save_fig = utils.define_path_save_file(
            path_save_file=path_save_fig)

        name_figure = 'confusion_matrix_display-{}.png'.format(
            utils.get_current_datetime())
        plt.savefig(
            ''.join([path_save_fig, name_figure]), bbox_inches='tight')
        print('Figure {} saved in {} directory.'.format(
            name_figure, path_save_fig))

    if display_figure:
        plt.show()

    plt.close()

    print('*****FIM CONFUSION MATRIX DISPLAY******\n')

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
        x(pd.DataFrame): Pandas DataFrame with numeric data to be treated.
        thresholds(np.arange): Thresholds to remove the columns.

    Returns:
        tuple: Tuple with the x, with features removed, and the results of execution.
    """
    results = list()
    for threshold in thresholds:
        x = pre_processing.delete_columns_with_low_variance(
            x=x, threshold=threshold, separate_numeric_columns=False)
        results.append(x.shape[1])
    return x, results


def __print_correlation_matrix_summarized(correlation_matrix: pd.DataFrame, lower_limit: float, upper_limit: float, to_latex: bool = False) -> None:
    """Print the correlation matrix summarized.

    Args:
        correlation_matrix(pd.DataFrame): Correlation matrix.
        lower_limit(float): Lower limit of the interval.
        upper_limit(float): Upper limit of the interval.
        to_latex(bool, optional): Flag to print the results in latex format. Defaults to False.
    """
    indexes = correlation_matrix.index
    columns = correlation_matrix.columns

    correlation_summarized_tmp = pd.DataFrame(columns=['Corr_Between'])
    correlation_summarized = pd.DataFrame(
        columns=['Atributo 1', 'Atributo 2', 'Correlação'])

    keyword = '-and-'
    for column in columns:
        for index in indexes:
            value = correlation_matrix.loc[index, column]
            if (((value < lower_limit) or (value > upper_limit)) and value != 1.0):
                if ((keyword.join([index, column]) not in correlation_summarized_tmp['Corr_Between'].values) and (keyword.join([column, index]) not in correlation_summarized_tmp['Corr_Between'].values)):
                    correlation_summarized_tmp = pd.concat([correlation_summarized_tmp, pd.DataFrame.from_records(
                        [
                            {
                                'Corr_Between': keyword.join([index, column])
                            }
                        ])])
                    correlation_summarized = pd.concat([correlation_summarized, pd.DataFrame.from_records(
                        [
                            {
                                'Atributo 1': index,
                                'Atributo 2': column,
                                'Correlação': value
                            }
                        ])])

    correlation_summarized = correlation_summarized.sort_values(
        by=['Correlação'], ascending=False)

    print('Correlation summarized:')
    displayhook(correlation_summarized)

    if to_latex:
        print('Correlation summarized in latex format:')
        print(correlation_summarized.to_latex(
            index=False, float_format="%.2f"))
