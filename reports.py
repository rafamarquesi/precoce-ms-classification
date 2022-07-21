import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import dataframe_image as dfi


def print_informations(data_frame: pd.DataFrame) -> None:
    """Print some informations of the DataFrame.

    Args:
        data_frame (pd.DataFrame): DataFrame to be printed.
    """

    print('\n*****INICIO PRINT INFOS******')
    # self.relatorio_atributos_base_dados(dados=dados_temp)
    print('Número total de linhas do DataFrame: {}'.format(len(data_frame.index)))
    print('Número de colunas: {}'.format(data_frame.columns.size))
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
            [{'Coluna': column, 'total_NaN': data_frame[column].isna().sum()}])])
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
    # print('Linhas duplicadas: {}'.format(rows_duplicated))
    print('Data Frame do Atríbuto {} duplicado: {}'.format(
        attribute, rows_duplicated))
    print('Relatório das colunas que divergem, entre os registros que tem o atributo {} igual.'.format(attribute))
    for id in rows_duplicated[attribute].unique():
        print('{}:{}'.format(attribute, id))
        rows_duplicated = data_frame.loc[data_frame[attribute] == id]
        for column_name, column_data in rows_duplicated.iteritems():
            comparison = column_data.ne(
                column_data.shift().bfill()).astype(int).values
            if not np.all(comparison == comparison[0]):
                print('Nome coluna que diverge: {}'.format(column_name))
                print('Index das linhas e valor na coluna que diverge:')
                print(column_data.ne(column_data.shift().bfill()).astype(int))
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


def class_distribution(y: np.ndarray) -> None:
    """Report of class distribution.

    Args:
        y (np.ndarray): NumPy Array to be treated.
    """
    print('\n*****INICIO RELATÓRIO DISTRIBUIÇÃO DE CLASES******')
    dist_1 = y[y.nonzero()].size / y.size
    dist_0 = (y.size - y[y.nonzero()].size) / y.size
    print('Distribuição da classe 1: {0:.0%}'.format(dist_1))
    print('Distribuição da classe 0: {0:.0%}'.format(dist_0))
    if dist_1 > dist_0:
        print('Erro majoritário: {0:.0%}'.format(1 - dist_1))
    else:
        print('Erro majoritário: {0:.0%}'.format(1 - dist_0))
    print('*****FIM RELATÓRIO DISTRIBUIÇÃO DE CLASES******')


def magnify():
    return [dict(selector="th", props=[("font-size", "7pt")]),
            dict(selector="td", props=[('padding', "0em 0em")]),
            dict(selector="th:hover", props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover", props=[('max-width', '200px'), ('font-size', '12pt')])]


def correlation_matrix(data_frame: pd.DataFrame, method: str, attribute: str = None):
    """Create a correlation matrix from the DataFrame.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        method (str): Method to be used to create the correlation matrix.
        attribute (str): Attribute to be used to create the correlation matrix.
    """
    print('\n*****INICIO CORRELATION MATRIX******')
    if attribute is None:
        correlation_matrix = data_frame.corr(method=method)
        # sns.heatmap(correlation_matrix, xticklabels=correlation_matrix.columns,
        #             yticklabels=correlation_matrix.columns)
        # plt.show()

        cmap = sns.diverging_palette(5, 250, as_cmap=True)

        styled_table = correlation_matrix.style.background_gradient(cmap, axis=1)\
            .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
            .set_caption("Hover to magify")\
            .set_precision(2)\
            .set_table_styles(magnify())
        # TODO: Criar local para armarzenar as imagens
        dfi.export(styled_table, 'correlation_matrix.png', max_cols=-1)
    else:
        correlation_matrix = data_frame.corr(method=method)[attribute]
        # correlation_matrix.to_frame().style.background_gradient(
        #     cmap=sns.light_palette((260, 75, 60), input="husl", as_cmap=True))

    # print(correlation_matrix.to_frame().style.background_gradient(
    #     cmap=sns.light_palette((260, 75, 60), input="husl", as_cmap=True)))

    print('*****FIM CORRELATION MATRIX*********')
