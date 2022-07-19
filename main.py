import csv_treatments
import pre_processing
import reports

if __name__ == '__main__':
    csv_path = '/mnt/Dados/Mestrado_Computacao_Aplicada_UFMS/documentos_dissertacao/base_dados/TAB_MODELAGEM_RAFAEL_2020_1.csv'
    number_csv_lines = 50000
    label_encoder_columns_names = ['Maturidade']
    columns_label_encoded = {}

    # delete_columns_names = ['area so confinamento', 'Lista Trace', 'DataAbate_6m_ANT', 'data_homol_select', 'Frigorifico_CNPJ',
    #                         'Frigorifico_RazaoSocial', 'Motivo', 'data12m', 'data6m', 'data3m', 'data1m', 'data7d', 'med7d_formITUmax', 'med3m_formITUmax',
    #                         'dif_datas', 'tot12m_Chuva', 'med12m_TempInst', 'med12m_TempMin', 'med12m_UmidInst',
    #                         'med12m_NDVI', 'med12m_EVI', 'med6m_NDVI', 'med6m_EVI', 'med3m_NDVI', 'med3m_EVI',
    #                         'tot6m_Chuva', 'med6m_TempInst', 'med6m_UmidInst', 'med1m_NDVI', 'med1m_EVI',
    #                         'tot3m_Chuva', 'med3m_TempInst', 'med3m_UmidInst', 'med7d_NDVI', 'med7d_EVI',
    #                         'tot1m_Chuva', 'med1m_TempInst', 'med1m_UmidInst',
    #                         'tot7d_Chuva', 'med7d_TempInst', 'med7d_TempMin', 'med7d_UmidInst']

    precoce_ms_data_frame = csv_treatments.load_data(
        csv_path=csv_path, columns_names=None, number_csv_lines=number_csv_lines)

    precoce_ms_data_frame = pre_processing.delete_duplicate_rows_by_attribute(
        data_frame=precoce_ms_data_frame, attribute_name='ID_ANIMAL')

    precoce_ms_data_frame = pre_processing.delete_nan_rows(
        data_frame=precoce_ms_data_frame)

    reports.print_informations(precoce_ms_data_frame)

    precoce_ms_data_frame, columns_label_encoded = pre_processing.label_encoder_columns(
        data_frame=precoce_ms_data_frame, columns_label_encoded=columns_label_encoded, columns_names=label_encoder_columns_names)
