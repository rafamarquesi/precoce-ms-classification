import csv_treatments
import pre_processing

if __name__ == '__main__':
    csv_path = '/mnt/Dados/Mestrado_Computacao_Aplicada_UFMS/documentos_dissertacao/base_dados/TAB_MODELAGEM_RAFAEL_2020_1.csv'

    precoce_ms_data_frame = csv_treatments.load_data(
        csv_path=csv_path, columns_names=None, number_csv_lines=2000)

    precoce_ms_data_frame = pre_processing.delete_duplicate_rows_by_attribute(
        data_frame=precoce_ms_data_frame, attribute_name='ID_ANIMAL')

    precoce_ms_data_frame = pre_processing.delete_nan_rows(
        data_frame=precoce_ms_data_frame)
