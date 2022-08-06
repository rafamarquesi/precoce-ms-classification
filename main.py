# utilities
import csv_treatments
import pre_processing
import reports
import pattern_extraction
import numpy as np

# classification models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# TODO: Treat imbalanced classes (Book Albon - Chapter 5.5)

if __name__ == '__main__':

    csv_path = '/mnt/Dados/Mestrado_Computacao_Aplicada_UFMS/documentos_dissertacao/base_dados/TAB_MODELAGEM_RAFAEL_2020_1.csv'
    number_csv_lines = None

    # TODO: Verify if the label encoder is encoding the correct values, example: Maturidade 'd' -> 0, '2' -> 1, '4' -> 2 ...
    label_encoder_columns_names = [
        'Maturidade', 'Acabamento', 'QuestionarioClassificacaoEstabel', 'CATEGORIA', 'classificacao'
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

    # delete_columns_names = None
    delete_columns_names = [
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

    # Dictionary containing the instantiated classes of classifiers and the parameters for optimization.
    classifiers = {}
    models_results = {}

    execute_pre_processing = False
    execute_classifiers = False

    ######### CSV TREATMENTS #########

    precoce_ms_data_frame = csv_treatments.load_data(
        csv_path=csv_path, columns_names=delete_columns_names, number_csv_lines=number_csv_lines)

    precoce_ms_data_frame = csv_treatments.move_cloumns_last_positions(
        data_frame=precoce_ms_data_frame, columns_names=['classificacao'])

    # reports.print_informations(data_frame=precoce_ms_data_frame)

    # reports.print_list_columns(data_frame=precoce_ms_data_frame)

    reports.all_attributes(data_frame=precoce_ms_data_frame)

    ######### PRE PROCESSING #########
    if execute_pre_processing:

        precoce_ms_data_frame = pre_processing.delete_duplicate_rows_by_attribute(
            data_frame=precoce_ms_data_frame, attribute_name='ID_ANIMAL')

        precoce_ms_data_frame = pre_processing.delete_columns(
            data_frame=precoce_ms_data_frame, columns_names=['ID_ANIMAL'])

        precoce_ms_data_frame = pre_processing.delete_nan_rows(
            data_frame=precoce_ms_data_frame)

        reports.print_informations(precoce_ms_data_frame)

        precoce_ms_data_frame, columns_label_encoded = pre_processing.label_encoder_columns(
            data_frame=precoce_ms_data_frame, columns_label_encoded=columns_label_encoded, columns_names=label_encoder_columns_names)

        precoce_ms_data_frame, columns_one_not_encoded = pre_processing.one_hot_encoder_columns(
            data_frame=precoce_ms_data_frame, columns_one_not_encoded=columns_one_not_encoded, columns_names=one_hot_encoder_columns_names)

        precoce_ms_data_frame, columns_min_max_scaled = pre_processing.min_max_scaler_columns(
            data_frame=precoce_ms_data_frame, columns_min_max_scaled=columns_min_max_scaled, columns_names=min_max_scaler_columns_names)

        reports.correlation_matrix(
            data_frame=precoce_ms_data_frame, method='pearson', attribute='classificacao',
            display_matrix=False, export_matrix=False, path_save_matrix='./plots')

        precoce_ms_data_frame = pre_processing.drop_feature_by_correlation(
            data_frame=precoce_ms_data_frame, method='pearson', columns_names=['Maturidade', 'Acabamento', 'Peso', 'classificacao'])

    ######### PATTERN EXTRACTION #########
    if execute_classifiers:

        x, y = pattern_extraction.create_x_y_data(
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
        print(models_results)
