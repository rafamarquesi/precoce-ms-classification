# import sys

# utilities
import csv_treatments
import pre_processing
import reports
import pattern_extraction
import utils
import numpy as np
import pandas as pd

# classification models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_rows', 5000)

# TODO: Treat imbalanced classes (Book Albon - Chapter 5.5)

# TODO: Use the cross_validate function, for evaluation of multiple metrics (https://scikit-learn.org/stable/modules/cross_validation.html#multimetric-cross-validation)

if __name__ == '__main__':

    # # Create a log file
    # run_log_file = open(
    #     'logs/run_log-{}.txt'.format(utils.get_current_datetime()), 'w'
    # )
    # sys.stdout = run_log_file

    # Path to the dataset
    csv_path = '/mnt/Dados/Mestrado_Computacao_Aplicada_UFMS/documentos_dissertacao/base_dados/TAB_MODELAGEM_RAFAEL_2020_1.csv'

    # Number of lines to be read from the dataset
    number_csv_lines = 100000

    # Dictionay with type of data for each column
    dtype_dict = {
        'ID_ANIMAL': 'uint32',
        'EstabelecimentoMunicipio': 'category',
        'Frigorifico_ID': 'uint8',
        'Frigorifico_CNPJ': 'uint64',
        'Frigorifico_RazaoSocial': 'category',
        'Municipio_Frigorifico': 'category',
        'Tipificacao': 'category',
        'Maturidade': 'category',
        'Acabamento': 'category',
        'Peso': 'float32',
        'EstabelecimentoIdentificador': 'uint16',
        'Questionario_ID': 'uint16',
        'QuestionarioClassificacaoEstabel': 'uint8',
        'FERTIIRRIGACAO': 'uint8',
        'ILP': 'uint8',
        'IFP': 'uint8',
        'ILPF': 'uint8',
        'CONCEN_VOLUM': 'UInt8',
        'CREEPFEEDING': 'UInt8',
        'FORN_ESTRAT_SILAGEM': 'UInt8',
        'PROTEICO': 'UInt8',
        'PROTEICO_ENERGETICO': 'UInt8',
        'RACAO_BAL_CONS_INFERIOR': 'UInt8',
        'SAL_MINERAL': 'UInt8',
        'SALMINERAL_UREIA': 'UInt8',
        'RACAOO_BAL_CONSUMO_IG': 'UInt8',
        'GRAO_INTEIRO': 'UInt8',
        'ALTO_CONCENTR_VOLUM': 'UInt8',
        'ALTO_CONCENTRADO': 'UInt8',
        'QuestionarioPossuiOutrosIncentiv': 'uint8',
        'QuestionarioFabricaRacao': 'uint8',
        'area so confinamento': 'UInt8',
        'regua de manejo': 'UInt8',
        'boa cobertura vegetal, com baixa': 'UInt8',
        'erosaoo laminar ou em sulco igua': 'UInt8',
        'identificacao individual': 'UInt8',
        'rastreamento SISBOV': 'UInt8',
        'Lista Trace': 'UInt8',
        'BPA': 'UInt8',
        'participa de aliancas mercadolog': 'UInt8',
        'QuestionarioPraticaRecuperacaoPa': 'uint8',
        'Confinamento': 'UInt8',
        'Suplementacao_a_campo': 'UInt8',
        'SemiConfinamento': 'UInt8',
        'dif_datas': 'uint16',
        'tot7d_Chuva': 'float32',
        'med7d_TempInst': 'float32',
        'med7d_TempMin': 'float32',
        'med7d_UmidInst': 'float32',
        'med7d_formITUinst': 'float32',
        'med7d_formITUmax': 'float32',
        'med7d_NDVI': 'float32',
        'med7d_EVI': 'float32',
        'med7d_preR_soja': 'float32',
        'med7d_preR_milho': 'float32',
        'med7d_preR_boi': 'float32',
        'tot1m_Chuva': 'float32',
        'med1m_TempInst': 'float32',
        'med1m_UmidInst': 'float32',
        'med1m_formITUinst': 'float32',
        'med1m_NDVI': 'float32',
        'med1m_EVI': 'float32',
        'med1m_preR_soja': 'float32',
        'med1m_preR_milho': 'float32',
        'med1m_preR_boi': 'float32',
        'tot3m_Chuva': 'float32',
        'med3m_TempInst': 'float32',
        'med3m_UmidInst': 'float32',
        'med3m_formITUinst': 'float32',
        'med3m_formITUmax': 'float32',
        'med3m_NDVI': 'float32',
        'med3m_EVI': 'float32',
        'med3m_preR_soja': 'float32',
        'med3m_preR_milho': 'float32',
        'med3m_preR_boi': 'float32',
        'tot6m_Chuva': 'float32',
        'med6m_TempInst': 'float32',
        'med6m_UmidInst': 'float32',
        'med6m_formITUinst': 'float32',
        'med6m_NDVI': 'float32',
        'med6m_EVI': 'float32',
        'med6m_preR_soja': 'float32',
        'med6m_preR_milho': 'float32',
        'med6m_preR_boi': 'float32',
        'tot12m_Chuva': 'float32',
        'med12m_TempInst': 'float32',
        'med12m_TempMin': 'float32',
        'med12m_UmidInst': 'float32',
        'med12m_formITUinst': 'float32',
        'med12m_NDVI': 'float32',
        'med12m_EVI': 'float32',
        'med12m_preR_soja': 'float32',
        'med12m_preR_milho': 'float32',
        'med12m_preR_boi': 'float32',
        'cnt7d_CL_ITUinst': 'float32',
        'cnt1m_CL_ITUinst': 'float32',
        'cnt3m_CL_ITUinst': 'float32',
        'cnt6m_CL_ITUinst': 'float32',
        'cnt12m_CL_ITUinst': 'float32',
        'ANO': 'uint16',
        'CATEGORIA': 'category',
        'classificacao': 'category',
        'Motivo': 'category'
    }

    # List with dates to parse
    parse_dates = [
        'DataAbate', 'Data_homol', 'DataAbate_6m_ANT',
        'data_homol_select', 'data12m', 'data6m',
        'data3m', 'data1m', 'data7d'
    ]

    # List with columns to delete when loading dataset
    # delete_columns_names_on_load_data = None
    delete_columns_names_on_load_data = [
        'Frigorifico_ID', 'Frigorifico_CNPJ', 'Frigorifico_RazaoSocial', 'Municipio_Frigorifico',
        'EstabelecimentoIdentificador', 'Data_homol', 'Questionario_ID',
        'area so confinamento', 'Lista Trace', 'Motivo', 'data_homol_select', 'dif_datas',
        'DataAbate_6m_ANT', 'data12m', 'data6m', 'data3m', 'data1m', 'data7d',
        'tot7d_Chuva', 'med7d_TempInst', 'med7d_TempMin', 'med7d_UmidInst', 'med7d_formITUmax', 'med7d_NDVI', 'med7d_EVI',
        'tot1m_Chuva', 'med1m_TempInst', 'med1m_UmidInst', 'med1m_NDVI', 'med1m_EVI',
        'tot3m_Chuva', 'med3m_TempInst', 'med3m_UmidInst', 'med3m_formITUmax', 'med3m_NDVI', 'med3m_EVI',
        'tot6m_Chuva', 'med6m_TempInst', 'med6m_UmidInst', 'med6m_NDVI', 'med6m_EVI',
        'tot12m_Chuva', 'med12m_TempInst', 'med12m_TempMin', 'med12m_UmidInst', 'med12m_NDVI', 'med12m_EVI',
    ]

    # Dictionary containing the instantiated classes of classifiers and the parameters for optimization
    classifiers = dict()

    # Dictionary containing the execution results of the models
    models_results = dict()

    print_informations_dataset = True
    execute_pre_processing = False
    execute_classifiers = False

    ################################################## CSV TREATMENTS ##################################################

    # Load the dataset
    precoce_ms_data_frame = csv_treatments.load_data(
        csv_path=csv_path, delete_columns_names=delete_columns_names_on_load_data,
        number_csv_lines=number_csv_lines, dtype_dict=dtype_dict, parse_dates=parse_dates
    )

    if print_informations_dataset:
        # reports.print_list_columns(data_frame=precoce_ms_data_frame)

        # Print a report of all attributes
        reports.all_attributes(data_frame=precoce_ms_data_frame)

        # Delete the duplicated rows by attribute, and print the report
        precoce_ms_data_frame = pre_processing.delete_duplicate_rows_by_attribute(
            data_frame=precoce_ms_data_frame, attribute_name='ID_ANIMAL', print_report=True)

        precoce_ms_data_frame = utils.delete_columns(
            data_frame=precoce_ms_data_frame, columns_names=['ID_ANIMAL'])

        # Delete NaN rows
        precoce_ms_data_frame = pre_processing.delete_nan_rows(
            data_frame=precoce_ms_data_frame, print_report=True)

        # Print the unique values for each column
        reports.unique_values_for_each_column(
            data_frame=precoce_ms_data_frame
        )

        # Print the percentage of unique values for each column
        reports.percentage_unique_values_for_each_column(
            data_frame=precoce_ms_data_frame, threshold=1
        )

        # Simulate delete columns with low variance
        reports.simulate_delete_columns_with_low_variance(
            data_frame=precoce_ms_data_frame, thresholds=np.arange(
                0.0, 0.55, 0.05),
            separate_numeric_columns=True)

        # TODO: Calculate feature importance with python, after encode categorical features (https://machinelearningmastery.com/calculate-feature-importance-with-python/)

    ################################################## PRE PROCESSING ##################################################

    if execute_pre_processing:
        # Identify columns that contain a single value, and delete them
        pre_processing.delete_columns_with_single_value(
            data_frame=precoce_ms_data_frame
        )

        # TODO: Verify how implement function to delete columns with low variance in pre processing
        # pre_processing.delete_columns_with_low_variance(

        # Delete duplicated rows by attribute
        precoce_ms_data_frame = pre_processing.delete_duplicate_rows_by_attribute(
            data_frame=precoce_ms_data_frame, attribute_name='ID_ANIMAL')

        # Delete column by names
        precoce_ms_data_frame = utils.delete_columns(
            data_frame=precoce_ms_data_frame, columns_names=['ID_ANIMAL'])

        # Delete NaN rows
        precoce_ms_data_frame = pre_processing.delete_nan_rows(
            data_frame=precoce_ms_data_frame)

        reports.informations(precoce_ms_data_frame)

        path_save_csv_after_pre_processing = '/mnt/Dados/Mestrado_Computacao_Aplicada_UFMS/documentos_dissertacao/base_dados/TAB_MODELAGEM_RAFAEL_2020_1_after_pre_processing-{}.csv'.format(
            utils.get_current_datetime())

        ordinal_encoder_columns_names = {
            'Maturidade': ['d', '2', '4', '6', '8'],
            'Acabamento': [
                'Magra - Gordura Ausente',
                'Gordura Escassa - 1 A 3 Mm De Espessura',
                'Gordura Mediana - Acima De 3 A Até 6 Mm De Espessura',
                'Gordura Uniforme - Acima De 6 E Até 10 Mm De Espessura',
                'Gordura Excessiva - Acima De 10 Mm De Espessura'
            ],
            'QuestionarioClassificacaoEstabel': ['0', '21', '26', '30'],
            'CATEGORIA': ['D', 'C', 'BB', 'BBB', 'AA', 'AAA']
        }
        columns_ordinal_encoded = {}

        label_encoder_columns_names = [
            'classificacao'
        ]
        columns_label_encoded = {}

        one_hot_encoder_columns_names = [
            'EstabelecimentoMunicipio', 'DataAbate', 'Tipificacao', 'ANO'
        ]
        columns_one_not_encoded = {}

        min_max_scaler_columns_names = [
            'Peso',
            'med7d_formITUinst', 'med7d_preR_soja', 'med7d_preR_milho', 'med7d_preR_boi',
            'med1m_formITUinst', 'med1m_preR_soja', 'med1m_preR_milho', 'med1m_preR_boi',
            'med3m_formITUinst', 'med3m_preR_soja', 'med3m_preR_milho', 'med3m_preR_boi',
            'med6m_formITUinst', 'med6m_preR_soja', 'med6m_preR_milho', 'med6m_preR_boi',
            'med12m_formITUinst', 'med12m_preR_soja', 'med12m_preR_milho', 'med12m_preR_boi'
        ]
        columns_min_max_scaled = {}

        precoce_ms_data_frame, columns_ordinal_encoded = pre_processing.ordinal_encoder_columns(
            data_frame=precoce_ms_data_frame, columns_ordinal_encoded=columns_ordinal_encoded, columns_names=ordinal_encoder_columns_names)

        precoce_ms_data_frame, columns_label_encoded = pre_processing.label_encoder_columns(
            data_frame=precoce_ms_data_frame, columns_label_encoded=columns_label_encoded, columns_names=label_encoder_columns_names)

        precoce_ms_data_frame, columns_one_not_encoded = pre_processing.one_hot_encoder_columns(
            data_frame=precoce_ms_data_frame, columns_one_hot_encoded=columns_one_not_encoded, columns_names=one_hot_encoder_columns_names)

        precoce_ms_data_frame, columns_min_max_scaled = pre_processing.min_max_scaler_columns(
            data_frame=precoce_ms_data_frame, columns_min_max_scaled=columns_min_max_scaled, columns_names=min_max_scaler_columns_names)

        # TODO: Spearman's Correlation, Further, the two variables being considered may have a non-Gaussian distribution. (https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/)
        reports.correlation_matrix(
            data_frame=precoce_ms_data_frame, method='pearson', attribute='classificacao',
            display_matrix=False, export_matrix=True, path_save_matrix='./plots')

        precoce_ms_data_frame = pre_processing.drop_feature_by_correlation(
            data_frame=precoce_ms_data_frame, method='pearson', columns_names=['Maturidade', 'Acabamento', 'Peso', 'classificacao'])

        precoce_ms_data_frame = csv_treatments.move_cloumns_last_positions(
            data_frame=precoce_ms_data_frame, columns_names=['classificacao'])

        csv_treatments.generate_new_csv(
            data_frame=precoce_ms_data_frame, csv_path=path_save_csv_after_pre_processing)

    ################################################## PATTERN EXTRACTION ##################################################

    if execute_classifiers:

        x, y = utils.create_x_y_numpy_data(
            data_frame=precoce_ms_data_frame)

        print('\nX: ', type(x))
        print('Y: ', type(y))

        reports.class_distribution(y)

        # TODO: Implement XGBoost and TabNet
        # k-nearest neighbors
        # Algorithm parameter{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’ for now is in auto
        # ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.
        # leaf_sizeint, default=30
        # Is the best option?
        # Varying the k neighbors parameter, we can find the best k value for the classification.
        param_dist = {
            'n_neighbors': list(np.arange(3, 20, 2)),
            'metric': ['euclidean'],
            'weights': ['uniform', 'distance']
        }
        classifiers[KNeighborsClassifier().__class__.__name__] = [
            KNeighborsClassifier(), param_dist]

        # Naive bayes
        # Algorithm to be beaten. Used as base.
        # It's generating low values. Is it because the attributes are not discrete (words)?
        # For non-discrete attributes, should I use another algorithm?
        param_dist = {}
        classifiers[GaussianNB().__class__.__name__] = [
            GaussianNB(), param_dist]

        # # Decision Trees (c4.5)
        # # Algorithm parameter settings, look for more (see documentation):
        # # min_impurity_decrease ??? The algorithm presents an error when a value greater than 0.0 is added
        # # ccp_alpha ??? the algorithm has an error when increasing the value
        # # max_depth -> use a great search to calibrate
        # # class_weight
        # # I identified, that for the default parameters, the maximum depth of the created tree is 9. For the current amount of data.
        # # criterion{“gini”, “entropy”},
        # param_dist = {
        #     'criterion': ['gini', 'entropy'],
        #     'max_depth': list(np.arange(1, 11)) + [None],
        #     'random_state': [0],
        #     'class_weight': ['balanced', None]
        # }
        # classifiers[DecisionTreeClassifier().__class__.__name__] = [
        #     DecisionTreeClassifier(), param_dist]

        # # Neural Network
        # # param_dist = {'solver': ['sgd'], 'learning_rate' : ['constant'], 'momentum' : scipy.stats.expon(scale=.1),
        # # 'alpha' : scipy.stats.expon(scale=.0001), 'activation' : ['logistic'],
        # # 'learning_rate_init' : scipy.stats.expon(scale=.01), 'hidden_layer_sizes':(200,100), 'max_iter':[500]}
        # # learning_rate_init -> change this parameter if the result is not good
        # # max_iter -> can also help to improve the result
        # # hidden_layer_sizes -> (layer_x_with_y_neurons, layer_x_with_y_neurons)
        # param_dist = {
        #     'solver': ['adam'],
        #     'learning_rate': ['constant'],
        #     'alpha': [0.001],
        #     'activation': ['relu'],
        #     'hidden_layer_sizes': (200, 100),
        #     'max_iter': [1000]
        # }
        # classifiers[MLPClassifier().__class__.__name__] = [
        #     MLPClassifier(), param_dist]

        # # Vector Support Machine
        # # kernel: ‘linear’, ‘poly’, ‘rbf’
        # #     # C: 10^x (-2 a 2)
        # #     # 'max_iter': [100, 1000]
        # #     # gamma : auto
        # param_dist = {
        #     'kernel': ['linear', 'poly', 'rbf'],
        #     'C': list(np.power(10, np.arange(-2, 2, dtype=np.float16))),
        #     'max_iter': [10000],
        #     'gamma': ['auto']
        # }
        # classifiers[SVC().__class__.__name__] = [SVC(), param_dist]

        # # Random forest classifier
        # param_dist = {
        #     'n_estimators': [10, 50, 100],
        #     'criterion': ['gini', 'entropy'],
        #     'max_depth': list(np.arange(1, 11)) + [None],
        #     'random_state': [0],
        #     'class_weight': ['balanced', None]
        # }
        # classifiers[RandomForestClassifier().__class__.__name__] = [
        #     RandomForestClassifier(), param_dist]

        # Running the classifiers
        models_results = pattern_extraction.run_models(
            x=x, y=y, models=classifiers, models_results=models_results)

        reports.models_results(
            models_results=models_results, path_save_fig='./plots')

    # run_log_file.close()
