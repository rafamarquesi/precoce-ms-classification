from typing import Union

import reports
import utils
import pre_processing

import pandas as pd
import numpy as np

pd.set_option('float_format', '{:f}'.format)


@utils.timeit
def load_data(csv_path: str, sep: str = ';', encoding: str = 'latin1', decimal: str = ',', delete_columns_names: list = None, number_csv_lines: int = None, dtype_dict: dict = None, parse_dates: list = False) -> pd.DataFrame:
    """Load the CSV file, given the path, column names, and number of rows to load, passed as parameters.

    Args:
        csv_path (str): Path where the CSV file is located.
        str (str, optional): Separator of the CSV file. Defaults to ';'.
        encoding (str, optional): Encoding of the CSV file. Defaults to 'latin1'.
        decimal (str, optional): Decimal separator of the CSV file. Defaults to ','.
        delete_columns_names (list, optional): Array of strings with columns names to exclude. Defaults to None.
        number_csv_lines (int, optional): Number of lines that will be loaded from the CSV file. Defaults to None.
        dtype_dict (dict, optional): Dictionary with the types of the columns. Defaults to None.
        parse_dates (list, optional): Array of strings with columns names to parse as dates. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame with the data loaded from the CSV file.
    """

    print('\n*****INICIO LOAD DATA******')
    if number_csv_lines != None:
        temp_data = pd.read_csv(
            csv_path, sep=sep, encoding=encoding, decimal=decimal,
            nrows=number_csv_lines, dtype=dtype_dict, parse_dates=parse_dates
        )
    else:
        temp_data = pd.read_csv(
            csv_path, sep=sep, encoding=encoding, decimal=decimal, dtype=dtype_dict, parse_dates=parse_dates
        )

    if ('Tipificacao' in temp_data.columns) and ('Fï¿½mea' in temp_data['Tipificacao'].array.categories):
        temp_data['Tipificacao'] = temp_data['Tipificacao'].map(
            {
                'Fï¿½mea': 'Fêmea',
                'Macho Castrado': 'Macho Castrado',
                'Macho Inteiro': 'Macho Inteiro'
            }
        )

    if 'CATEGORIA_BINARIA' in temp_data.columns:
        temp_data['CATEGORIA_BINARIA'] = temp_data['CATEGORIA_BINARIA'].map(
            {'0': 'Baixa qualidade', '1': 'Alta qualidade'}
        )

    reports.informations(temp_data)
    # Used to print min and max of each column
    # reports.min_max_column(temp_data)

    if delete_columns_names != None:
        temp_data = utils.delete_columns(
            data_frame=temp_data, delete_columns_names=delete_columns_names
        )

        reports.informations(temp_data)

    print('*****FIM LOAD DATA******')

    return temp_data


@utils.timeit
def generate_new_csv(data_frame: pd.DataFrame, csv_path: str, csv_name: str, sep: str = ';', encoding: str = 'latin1', index: bool = False, date_format: str = '%d%b%Y', decimal: str = ',') -> None:
    """Generate a new CSV file, given the DataFrame and the path, passed as parameters.

    Args:
        data_frame (pd.DataFrame): DataFrame to be treated.
        csv_path (str): Path where the CSV file will be generated.
        csv_name (str): Name of the CSV file to be generated.
        sep (str, optional): Separator of the CSV file. Defaults to ';'.
        encoding (str, optional): Encoding of the CSV file. Defaults to 'latin1'.
        index (bool, optional): If True, the index will be included in the CSV file. Defaults to False.
        date_format (str, optional): Date format of the CSV file. Defaults to '%d%b%Y'.
        decimal (str, optional): Decimal separator of the CSV file. Defaults to ','.
    """
    csv_path = utils.define_path_save_file(
        path_save_file=csv_path) + csv_name + '.csv'

    data_frame.to_csv(
        csv_path, sep=sep, encoding=encoding,
        index=index, date_format=date_format,
        decimal=decimal
    )
    print('\nCSV gerado com sucesso em: {}.'.format(csv_path))


def choose_csv_path(sampling: str = None, folder_path: str = None) -> str:
    """Choose the path of the CSV file to be loaded.

    Args:
        sampling (str, optional): Key name of the CSV file to be loaded. The sampling keys availables to load are '0.2', '0.5', '2', '5', '10', '20', '30', '40', '50', '60', '100', and 'after_correlation-02-11-2022_18:29:05'. Defaults to None.
        folder_path (str, optional): Path of the folder where the CSV file is located. Defaults to None.

    Returns:
        str: Path of the CSV file to be loaded.
    """

    folder_path = utils.define_path_save_file(folder_path)

    csv_files = {
        '0.2': '{}TAB_MODELAGEM_RAFAEL_2020_1-0.2-percentage-sampling.csv'.format(folder_path),
        '0.5': '{}TAB_MODELAGEM_RAFAEL_2020_1-0.5-percentage-sampling.csv'.format(folder_path),
        '2': '{}TAB_MODELAGEM_RAFAEL_2020_1-2.0-percentage-sampling.csv'.format(folder_path),
        '5': '{}TAB_MODELAGEM_RAFAEL_2020_1-5.0-percentage-sampling.csv'.format(folder_path),
        '10': '{}TAB_MODELAGEM_RAFAEL_2020_1-10.0-percentage-sampling.csv'.format(folder_path),
        '20': '{}TAB_MODELAGEM_RAFAEL_2020_1-20.0-percentage-sampling.csv'.format(folder_path),
        '30': '{}TAB_MODELAGEM_RAFAEL_2020_1-30.0-percentage-sampling.csv'.format(folder_path),
        '40': '{}TAB_MODELAGEM_RAFAEL_2020_1-40.0-percentage-sampling.csv'.format(folder_path),
        '50': '{}TAB_MODELAGEM_RAFAEL_2020_1-50.0-percentage-sampling.csv'.format(folder_path),
        '60': '{}TAB_MODELAGEM_RAFAEL_2020_1-60.0-percentage-sampling.csv'.format(folder_path),
        '100': '{}TAB_MODELAGEM_RAFAEL_2020_1.csv'.format(folder_path),
        'after_correlation-05-11-2022_18:16:37': '{}TAB_MODELAGEM_RAFAEL_2020_1-after_drop_feature_by_correlation-05-11-2022_18:16:37.csv'.format(folder_path),
        'ANIMAIS-POR-LOTE-CATEGORIA_BINARIA': '{}TAB_MODELAGEM_RAFAEL_2020_1-ANIMAIS-POR-LOTE-CATEGORIA_BINARIA.csv'.format(folder_path),
    }

    if sampling not in csv_files.keys():
        raise ValueError(
            'The sampling parameter must be one of values: {}.'.format(csv_files.keys()))

    return csv_files.get(sampling)


@utils.timeit
def generate_batch_dataset(precoce_ms_data_frame: pd.DataFrame, attrs_groupby: list, folder_path: str, csv_name: str) -> None:
    """Creates a new dataset from the original dataset by batching the animals. The batch is grouped by the attributes passed in the parameter attrs_groupby.

    Args:
        precoce_ms_data_frame (pd.DataFrame): DataFrame to be treated.
        attrs_groupby (list): List of attributes to be used to group the animals.
        folder_path (str): Path where the CSV file will be generated.
        csv_name (str): Name of the CSV file to be generated.
    """
    print('\n*****INICIO GENERATE BATCH DATASET******')

    # Delete duplicated rows by attribute
    precoce_ms_data_frame = pre_processing.delete_duplicate_rows_by_attribute(
        data_frame=precoce_ms_data_frame, attribute_name='ID_ANIMAL')

    # Group the data frame by the attributes passed in the parameter attrs_groupby
    df_grouped = precoce_ms_data_frame.groupby(attrs_groupby)

    agg_dict = {
        'ID_ANIMAL': 'count',  # The sum of the animals in the batch
        'Peso': 'mean',  # The average weight of the animals in the batch
        'QuestionarioClassificacaoEstabel': __get_most_frequent_value,
        'ILP': __get_most_frequent_value,
        'IFP': __get_most_frequent_value,
        'ILPF': __get_most_frequent_value,
        'QuestionarioPossuiOutrosIncentiv': __get_most_frequent_value,
        'QuestionarioFabricaRacao': __get_most_frequent_value,
        'regua de manejo': __get_most_frequent_value,
        'identificacao individual': __get_most_frequent_value,
        'rastreamento SISBOV': __get_most_frequent_value,
        'participa de aliancas mercadolog': __get_most_frequent_value,
        'QuestionarioPraticaRecuperacaoPa': __get_most_frequent_value,
        'Confinamento': __get_most_frequent_value,
        'Suplementacao_a_campo': __get_most_frequent_value,
        'SemiConfinamento': __get_most_frequent_value,
        'tot3m_Chuva': 'mean',
        'med3m_formITUinst': 'mean',
        'med3m_NDVI': 'mean',
        'med3m_preR_milho': 'mean',
        'med3m_preR_boi': 'mean',
        'CATEGORIA': __categorize_batch_into_two_categories
    }

    df_grouped_agg = None

    for key, value in agg_dict.items():
        print('Processing the attribute: {}'.format(key))
        # Agregate the data frame
        df_agg = df_grouped.agg(
            {
                key: value
            }
        )

        # Reset the index
        df_agg.reset_index(inplace=True)

        if df_grouped_agg is None:
            df_grouped_agg = df_agg
        else:
            df_grouped_agg = pd.merge(
                df_grouped_agg, df_agg, on=attrs_groupby, how='left')
            # df_grouped_agg = pd.concat(
            #     [df_grouped_agg, df_agg.iloc[:, -1:]], axis=1)

    print('Shape BEFORE drop rows that have only NaN values: {}'.format(
        df_grouped_agg.shape))

    # Delete the rows that have only NaN values
    index_to_drop = []
    for index in df_grouped_agg.index:
        if df_grouped_agg.loc[index, :].drop(labels=attrs_groupby+['ID_ANIMAL']).isna().all():
            index_to_drop.append(index)
    df_grouped_agg.drop(index_to_drop, axis=0, inplace=True)

    print('Shape AFTER drop rows that have only NaN values: {}'.format(
        df_grouped_agg.shape))

    df_grouped_agg.rename(
        columns={
            'ID_ANIMAL': 'QTD_ANIMAIS_LOTE',
            'Peso': 'PESO_MEDIO_LOTE',
            'CATEGORIA': 'CATEGORIA_BINARIA'
        },
        inplace=True
    )

    generate_new_csv(
        data_frame=df_grouped_agg,
        csv_path=folder_path,
        csv_name=csv_name
    )

    print('*****FIM GENERATE BATCH DATASET******')


################################################## PRIVATE METHODS ##################################################


def __categorize_batch_into_two_categories(attr_categoria: pd.Series) -> int:
    """It categorizes a batch of animals, between 1: for the batch that is mostly composed of animals categorized above quality ('AAA', 'AA'), and 0: for the batch that is mostly composed of animals categorized below quality, belonging to other categories ('BBB', 'BB', 'C', 'D').

    Args:
        attr_categoria (pd.Series): Series with the categories of the animals in the batch. The categories are: 'AAA', 'AA', 'BBB', 'BB', 'C', 'D'.

    Returns:
        int: 1 for the batch that is mostly composed of animals categorized above quality ('AAA', 'AA'), and 0 for the batch that is mostly composed of animals categorized below quality, belonging to other categories ('BBB', 'BB', 'C', 'D').
    """

    if attr_categoria.name != 'CATEGORIA':
        raise ValueError('The attribute must be "CATEGORIA"')

    categories_aaa_aa_sum = 0
    others_categories_sum = 0
    # for key, value in attr_categoria.value_counts().to_dict().items():
    for key, value in attr_categoria.value_counts().iteritems():
        if (key == 'AAA') or (key == 'AA'):
            categories_aaa_aa_sum += value
        else:
            others_categories_sum += value

    if categories_aaa_aa_sum >= others_categories_sum:
        return 1  # Acima da qualidade, o lote majoritariamente é composto por AAA e AA
    else:
        return 0  # Abaixo da qualidade, o lote majoritariamente é composto pelas outras categorias


def __get_most_frequent_value(attr: pd.Series) -> Union[int, str, None]:
    """It returns the index of the most frequent value of a Series.

    Args:
        attr (pd.Series): Series with the values.

    Returns:
        Union[int, str, None]: Index of the most frequent value of the Series.
    """
    try:
        return attr.value_counts().idxmax()
    except ValueError:
        return np.nan
