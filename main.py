import sys
import traceback

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import torch
from pytorch_tabnet.augmentations import ClassificationSMOTE

from clf_switcher import ClfSwitcher
from pytorch_tabnet_tuner.tab_model_tuner import TabNetClassifierTuner, F1ScoreMacro
from sklearn_tuner.model_selection_tuner import GridSearchCVTuner

import csv_treatments
import pre_processing
import reports
import pattern_extraction
import utils
import settings
from tee import Tee

# TODO: Treat imbalanced classes (Book Albon - Chapter 5.5) (https://github.com/alod83/data-science/blob/master/Preprocessing/Balancing/Balancing.ipynb) (https://machinelearningmastery.com/imbalanced-classification-with-the-adult-income-dataset/) (https://machinelearningmastery.com/imbalanced-classification-of-good-and-bad-credit/) (https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/)

# TODO: Use the cross_validate function, for evaluation of multiple metrics (https://scikit-learn.org/stable/modules/cross_validation.html#multimetric-cross-validation)

# TODO: IDEA: Places to run the code: https://www.linkedin.com/posts/ashishpatel2604_data-datascience-machinelearning-activity-6997071480664559616-doXm?utm_source=share&utm_medium=member_desktop

if __name__ == '__main__':
    try:
        # Create a log file
        tee_log_file = Tee(
            sys.stdout,
            open(
                ''.join(
                    [
                        utils.define_path_save_file(
                            path_save_file=settings.PATH_SAVE_LOGS),
                        'run_log-{}.log'.format(utils.get_current_datetime())
                    ]
                ),
                'w'
            )
        ).open()

        # Some settings are configured by default. If you want to change any settings,
        # just follow the instruction for the specific setting. For more information, view the settings.py file.

        # Number of jobs to run in parallel, where -1 means using all processors.
        settings.n_jobs = 1

        # Folder path where the CSV file is located
        settings.dataset_folder_path = '/mnt/Dados/Mestrado_Computacao_Aplicada_UFMS/documentos_dissertacao/base_dados/'

        # Path to the dataset
        settings.csv_path = csv_treatments.choose_csv_path(
            sampling='ANIMAIS-POR-LOTE-CATEGORIA_BINARIA', folder_path=settings.dataset_folder_path)

        # Number of lines to be read from the dataset, where None read all lines
        # settings.number_csv_lines = 1000

        # Class column name
        settings.class_column = 'CATEGORIA_BINARIA'

        # Checks if it is batch separated animals dataset
        # It was necessary to create to do some validations while loading the dataset, as it was changed from the original.
        is_batch_dataset = True if settings.csv_path.find(
            'ANIMAIS-POR-LOTE') != -1 else False

        # List with columns to delete when loading dataset
        settings.delete_columns_names_on_load_data = [
            'EstabelecimentoMunicipio', 'Frigorifico_ID', 'Frigorifico_CNPJ', 'Frigorifico_RazaoSocial', 'Municipio_Frigorifico',
            'Peso',
            'Maturidade', 'Acabamento',
            'EstabelecimentoIdentificador',
            'Questionario_ID', 'FERTIIRRIGACAO', 'CONCEN_VOLUM', 'CREEPFEEDING', 'FORN_ESTRAT_SILAGEM',
            'PROTEICO', 'PROTEICO_ENERGETICO', 'RACAO_BAL_CONS_INFERIOR', 'SAL_MINERAL', 'SALMINERAL_UREIA',
            'RACAOO_BAL_CONSUMO_IG', 'GRAO_INTEIRO', 'ALTO_CONCENTR_VOLUM', 'ALTO_CONCENTRADO', 'area so confinamento',
            # two columns above removed because they have 19.352582% of missing values
            'boa cobertura vegetal, com baixa', 'erosaoo laminar ou em sulco igua',
            'Lista Trace', 'BPA', 'dif_datas',
            'tot7d_Chuva', 'med7d_TempInst', 'med7d_TempMin', 'med7d_UmidInst',
            'med7d_formITUinst', 'med7d_formITUmax', 'med7d_NDVI', 'med7d_EVI',
            'med7d_preR_soja', 'med7d_preR_milho', 'med7d_preR_boi',
            'tot1m_Chuva', 'med1m_TempInst', 'med1m_UmidInst',
            'med1m_formITUinst', 'med1m_NDVI', 'med1m_EVI',
            'med1m_preR_soja', 'med1m_preR_milho', 'med1m_preR_boi',
            'med3m_TempInst', 'med3m_UmidInst', 'med3m_formITUmax', 'med3m_EVI', 'med3m_preR_soja',
            'tot6m_Chuva', 'med6m_TempInst', 'med6m_UmidInst', 'med6m_formITUinst', 'med6m_NDVI',
            'med6m_EVI', 'med6m_preR_soja', 'med6m_preR_milho', 'med6m_preR_boi',
            'tot12m_Chuva', 'med12m_TempInst', 'med12m_TempMin', 'med12m_UmidInst', 'med12m_formITUinst',
            'med12m_NDVI', 'med12m_EVI', 'med12m_preR_soja', 'med12m_preR_milho', 'med12m_preR_boi',
            'cnt7d_CL_ITUinst', 'cnt1m_CL_ITUinst', 'cnt3m_CL_ITUinst', 'cnt6m_CL_ITUinst', 'cnt12m_CL_ITUinst',
            'ANO', 'Motivo',
            'DataAbate',
            'Data_homol', 'DataAbate_6m_ANT', 'data_homol_select',
            'data12m', 'data6m', 'data3m', 'data1m', 'data7d',
            # column above removed because it will not have the attribute at the time of performing the prediction and the target is derived from this attribute
            'classificacao',
            'CATEGORIA'
        ]

        # Dict update for ordinal encoding
        # settings.ordinal_encoder_columns_names.update(
        #     {
        #         'DataAbate': None
        #     }
        # )
        settings.ordinal_encoder_columns_names.pop('CATEGORIA')
        settings.ordinal_encoder_columns_names.pop('Maturidade')
        settings.ordinal_encoder_columns_names.pop('Acabamento')

        # List with column names to apply the label encoder
        settings.label_encoder_columns_names = [
            settings.class_column
        ]

        # List with column names to apply the one hot encoder
        settings.one_hot_encoder_columns_names = [
            'Tipificacao'
        ]

        # List with column names to apply the min max scaler
        settings.min_max_scaler_columns_names = [
            'QTD_ANIMAIS_LOTE',
            'PESO_MEDIO_LOTE',
            'tot3m_Chuva', 'med3m_formITUinst', 'med3m_NDVI', 'med3m_preR_milho', 'med3m_preR_boi'
        ]

        # List with column names to apply the simple imputer
        # settings.simple_imputer_columns_names = [
        #     'rastreamento SISBOV', 'regua de manejo', 'identificacao individual',
        #     'participa de aliancas mercadolog', 'Confinamento', 'Suplementacao_a_campo',
        #     'SemiConfinamento'
        # ]

        # List with column names to drop feature by correlation
        # I choise the features greater than or equal to threshold 0.95, because the spearman correlation
        # matrix showed that there are some features that are highly correlated
        settings.columns_names_drop_feature_by_correlation = [
            'med3m_formITUinst', 'med3m_preR_boi',
            settings.class_column
        ]

        dataset_reports = False
        execute_pre_processing = False
        execute_classifiers = False
        execute_classifiers_pipeline = True
        analyze_results = False

        ################################################## CSV TREATMENTS ##################################################

        generate_samples = False
        generate_batch_dataset = False

        if generate_samples:
            # Generate sample of dataset
            precoce_ms_data_frame = csv_treatments.load_data(
                csv_path=settings.csv_path, number_csv_lines=settings.number_csv_lines,
                dtype_dict=settings.dtype_dict, parse_dates=settings.parse_dates
            )

            percentages = [0.002, 0.005, 0.02,
                           0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            for percentage in percentages:
                csv_treatments.generate_new_csv(
                    data_frame=utils.random_sampling_data(
                        data_frame=precoce_ms_data_frame, how_generate='percentage', frac=percentage
                    ),
                    csv_path='/mnt/Dados/Mestrado_Computacao_Aplicada_UFMS/documentos_dissertacao/base_dados',
                    csv_name='TAB_MODELAGEM_RAFAEL_2020_1-{}-percentage-sampling'.format(
                        percentage*100)
                )
        else:
            if is_batch_dataset:
                settings.parse_dates = ['DataAbate']
            # Load the dataset
            precoce_ms_data_frame = csv_treatments.load_data(
                csv_path=settings.csv_path, delete_columns_names=settings.delete_columns_names_on_load_data,
                number_csv_lines=settings.number_csv_lines, dtype_dict=settings.dtype_dict, parse_dates=settings.parse_dates
            )

        if generate_batch_dataset:

            csv_treatments.generate_batch_dataset(
                precoce_ms_data_frame=precoce_ms_data_frame,
                attrs_groupby=[
                    'DataAbate',
                    'EstabelecimentoIdentificador',
                    'Tipificacao'],
                folder_path=settings.dataset_folder_path,
                csv_name='TAB_MODELAGEM_RAFAEL_2020_1-ANIMAIS-POR-LOTE-CATEGORIA_BINARIA'
            )

        ################################################## REPORTS ##################################################

        if dataset_reports:
            # reports.print_list_columns(data_frame=precoce_ms_data_frame)

            # Print a report of all attributes
            reports.all_attributes(data_frame=precoce_ms_data_frame)

            # Delete the duplicated rows by attribute, and print the report
            precoce_ms_data_frame = pre_processing.delete_duplicate_rows_by_attribute(
                data_frame=precoce_ms_data_frame, attribute_name='ID_ANIMAL', print_report=True)

            precoce_ms_data_frame = utils.delete_columns(
                data_frame=precoce_ms_data_frame, delete_columns_names=['ID_ANIMAL'])

            # Strategies for NaN rows
            # if is_batch_dataset:

            #     # Show report for NaN rows
            #     reports.nan_attributes(
            #         data_frame=precoce_ms_data_frame, total_nan=precoce_ms_data_frame.isna().sum().sum())

            #     # Simple imputer for NaN rows, strategy mean
            #     precoce_ms_data_frame = pre_processing.simple_imputer_dataframe(
            #         data_frame=precoce_ms_data_frame,
            #         strategy='mean',
            #         columns=settings.min_max_scaler_columns_names,
            #         verbose=True
            #     )

            #     # Simple imputer for NaN rows, strategy most_frequent
            #     precoce_ms_data_frame = pre_processing.simple_imputer_dataframe(
            #         data_frame=precoce_ms_data_frame,
            #         columns=settings.simple_imputer_columns_names,
            #         strategy='most_frequent',
            #         verbose=True
            #     )

            # else:
            # Delete NaN rows
            precoce_ms_data_frame = pre_processing.delete_nan_rows(
                data_frame=precoce_ms_data_frame, print_report=True)

            # Convert pandas dtypes to numpy dtypes, some operations doesn't work with pandas dtype, for exemple, the XGBoost models
            precoce_ms_data_frame = utils.convert_pandas_dtype_to_numpy_dtype(
                data_frame=precoce_ms_data_frame, pandas_dtypes=[pd.UInt8Dtype()])

            # Print histogram for each attribute
            # reports.histogram(
            #     data_frame=precoce_ms_data_frame
            # )

            # Print histogram for each attribute grouped by target class
            # reports.histogram_grouped_by_target(
            #     data_frame=precoce_ms_data_frame, target=settings.class_column
            # )

            # Print boxplot for each attribute
            # reports.boxplot(data_frame=precoce_ms_data_frame)

            # Print boxplot for each attribute by target class
            # reports.boxplot_grouped_by_target(
            #     data_frame=precoce_ms_data_frame, target=settings.class_column)

            # Print an attribute's outiliers
            if is_batch_dataset:
                attr_name = 'PESO_MEDIO_LOTE'
            else:
                attr_name = 'Peso'
            reports.detect_outiliers_from_attribute(
                data_frame=precoce_ms_data_frame, attribute_name=attr_name)

            # Print the unique values for each column
            reports.unique_values_for_each_column(
                data_frame=precoce_ms_data_frame
            )

            # Print the percentage of unique values for each column
            reports.percentage_unique_values_for_each_column(
                data_frame=precoce_ms_data_frame, threshold=1
            )

            # Identify columns that contain a single value, and delete them
            precoce_ms_data_frame = pre_processing.delete_columns_with_single_value(
                data_frame=precoce_ms_data_frame
            )

            # Simulate delete columns with low variance, using VarianceThreshold from sklearn
            # reports.simulate_delete_columns_with_low_variance(
            #     data_frame=precoce_ms_data_frame,
            #     thresholds=np.arange(0.0, 0.10, 0.05),
            #     separate_numeric_columns=True,
            #     path_save_fig=settings.PATH_SAVE_PLOTS,
            #     display_figure=True
            # )

            # Apply ordinal encoder to the columns
            precoce_ms_data_frame, settings.columns_ordinal_encoded = pre_processing.ordinal_encoder_columns(
                data_frame=precoce_ms_data_frame, columns_ordinal_encoded=settings.columns_ordinal_encoded,
                columns_names=settings.ordinal_encoder_columns_names
            )

            # Apply label encoder to the columns
            precoce_ms_data_frame, settings.columns_label_encoded = pre_processing.label_encoder_columns(
                data_frame=precoce_ms_data_frame, columns_label_encoded=settings.columns_label_encoded,
                columns_names=settings.label_encoder_columns_names
            )

            # Apply one hot encoder to the columns
            precoce_ms_data_frame, settings.columns_one_hot_encoded = pre_processing.one_hot_encoder_columns(
                data_frame=precoce_ms_data_frame, columns_one_hot_encoded=settings.columns_one_hot_encoded,
                columns_names=settings.one_hot_encoder_columns_names
            )

            # Apply min max scaler to the columns
            precoce_ms_data_frame, settings.columns_min_max_scaled = pre_processing.min_max_scaler_columns(
                data_frame=precoce_ms_data_frame, columns_min_max_scaled=settings.columns_min_max_scaled,
                columns_names=settings.min_max_scaler_columns_names
            )

            # Move the target column to the last position in dataframe
            precoce_ms_data_frame = utils.move_cloumns_last_positions(
                data_frame=precoce_ms_data_frame, columns_names=[settings.class_column])

            # Target attribute distribution
            reports.class_distribution(
                y=precoce_ms_data_frame[settings.class_column].values)

            # Spearman's Correlation, Further, the two variables being considered may have a non-Gaussian distribution.
            # The coefficient returns a value between -1 and 1 that represents the limits of correlation
            # from a full negative correlation to a full positive correlation. A value of 0 means no correlation.
            # The value must be interpreted, where often a value below -0.5 or above 0.5 indicates a notable correlation,
            # and values below those values suggests a less notable correlation.

            export_matrix = True

            # Correlation matrix using pearson method, between all attributes
            reports.correlation_matrix(
                data_frame=precoce_ms_data_frame, method='pearson',
                display_matrix=True, export_matrix=export_matrix, path_save_matrix=settings.PATH_SAVE_PLOTS,
                print_corr_matrix_summarized=True)

            # Correlation matrix using pearson method, between all attributes and the class attribute
            reports.correlation_matrix(
                data_frame=precoce_ms_data_frame, method='pearson', attribute=settings.class_column,
                display_matrix=True, export_matrix=export_matrix, path_save_matrix=settings.PATH_SAVE_PLOTS)

            # Correlation matrix using spearman method, between all attributes
            reports.correlation_matrix(
                data_frame=precoce_ms_data_frame, method='spearman',
                display_matrix=True, export_matrix=export_matrix, path_save_matrix=settings.PATH_SAVE_PLOTS,
                print_corr_matrix_summarized=True)

            # Correlation matrix using spearman method, between all attributes and the class attribute
            reports.correlation_matrix(
                data_frame=precoce_ms_data_frame, method='spearman', attribute=settings.class_column,
                display_matrix=True, export_matrix=export_matrix, path_save_matrix=settings.PATH_SAVE_PLOTS)

            # Delete features by correlation
            precoce_ms_data_frame = pre_processing.drop_feature_by_correlation(
                data_frame=precoce_ms_data_frame, method='spearman', columns_names=settings.columns_names_drop_feature_by_correlation)

            # Calculate feature importance with linear models
            # reports.feature_importance_using_coefficients_of_linear_models(
            #     data_frame=precoce_ms_data_frame,
            #     models=['logistic_regression', 'linear_svc', 'sgd_classifier'],
            #     path_save_fig=settings.PATH_SAVE_PLOTS,
            #     display_figure=True,
            #     class_weight='balanced',
            #     n_jobs=settings.n_jobs
            # )

            # Calculate feature importance with tree based models
            # reports.feature_importance_using_tree_based_models(
            #     data_frame=precoce_ms_data_frame,
            #     models=['decision_tree_classifier',
            #             'random_forest_classifier', 'xgb_classifier'],
            #     path_save_fig=settings.PATH_SAVE_PLOTS,
            #     display_figure=True,
            #     class_weight='balanced',
            #     n_jobs=settings.n_jobs
            # )

            # TODO: It didn't work, study better how permutation importance works
            # Calculate feature importance using permutation importance
            # reports.feature_importance_using_permutation_importance(
            #     data_frame=precoce_ms_data_frame,
            #     models=['knneighbors_classifier', 'gaussian_nb'],
            #     n_repeats=3,
            #     n_jobs=settings.n_jobs,
            #     path_save_fig=settings.PATH_SAVE_PLOTS,
            #     display_figure=True
            # )

            # Simulate sequential feature selector
            # reports.simulate_sequential_feature_selector(
            #     data_frame=precoce_ms_data_frame,
            #     estimator=KNeighborsClassifier(),
            #     k_features=(3, (precoce_ms_data_frame.shape[1] - 1)),
            #     forward=True,
            #     floating=False,
            #     scoring='accuracy',
            #     cv=3,
            #     n_jobs=settings.n_jobs,
            #     save_fig=False,
            #     path_save_fig=settings.PATH_SAVE_PLOTS
            # )

            # Simulate recursive feature elimination wiht cross validation
            reports.simulate_recursive_feature_elimination_with_cv(
                data_frame=precoce_ms_data_frame,
                estimator=DecisionTreeClassifier(),
                scoring='accuracy',
                n_jobs=settings.n_jobs,
                save_fig=False,
                path_save_fig=settings.PATH_SAVE_PLOTS
            )

            # # Apply inverse ordinal encoder to the columns
            # precoce_ms_data_frame, settings.columns_ordinal_encoded = pre_processing.inverse_ordinal_encoder_columns(
            #     data_frame=precoce_ms_data_frame, columns_ordinal_encoded=settings.columns_ordinal_encoded)

            # # Apply inverse label encoder to the columns
            # precoce_ms_data_frame, settings.columns_label_encoded = pre_processing.inverse_label_encoder_columns(
            #     data_frame=precoce_ms_data_frame, columns_label_encoded=settings.columns_label_encoded)

            # # Apply inverse one hot encoder to the columns
            # precoce_ms_data_frame, settings.columns_one_hot_encoded = pre_processing.inverse_one_hot_encoder_columns(
            #     data_frame=precoce_ms_data_frame, columns_one_hot_encoded=settings.columns_one_hot_encoded)

            # # Apply inverse min max scaler to the columns
            # precoce_ms_data_frame, settings.columns_min_max_scaled = pre_processing.inverse_min_max_scaler_columns(
            #     data_frame=precoce_ms_data_frame, columns_min_max_scaled=settings.columns_min_max_scaled)

            # # Move the target column to the last position in dataframe
            # precoce_ms_data_frame = utils.move_cloumns_last_positions(
            #     data_frame=precoce_ms_data_frame, columns_names=[settings.class_column])

        ################################################## PIPELINE FOR CLASSIFICATION #####################################

        if execute_classifiers_pipeline:

            # Get the number of classes in the target column
            class_number = len(
                precoce_ms_data_frame[settings.class_column].value_counts())

            ##### Grid Search Settings #####
            # Flag to run the original scikit-learn Grid Search CV or the scikit-learn Tuner Grid Search CV (persisting the objects, results, during the execution of the pipeline).
            # Wheter True, the Grid Search CV Tuner will be used, otherwise the original scikit-learn Grid Search CV will be used.
            settings.run_grid_search_cv_tuner = True

            # Flag to save the results of each split in the pipeline execution, to be used in a possible new execution,
            # in case the execution is interrupted.
            # Used only if run_grid_search_cv_tuner = True and It works only if n_jobs = 1 (don't work in parallel).
            # If false, the results for already executed parameters will be loaded,
            # but the results for new executed parameters will not be saved.
            if settings.n_jobs != 1:
                settings.save_results_during_run = False

            # Whether True, the objects persisted in the path_objects_persisted_results_runs will be cleaned before the execution of the pipeline
            settings.new_run = True

            ##### XGBoost Settings #####
            # The tree method to use for training the model. 'gpu_hist' is recommended for GPU training. 'hist' is recommended for CPU training.
            settings.tree_method = 'hist'
            # Specify the learning task and the corresponding learning objective. 'binary:logistic' is for binary classification.
            if class_number > 2:
                settings.objective = 'multi:softmax'

            ##### Tab Net Settings #####
            # For parallelization of the training, set the number of workers to be used. If 0, the number of workers will be set to the number of available CPU cores.
            # torch.multiprocessing.set_sharing_strategy('file_system')
            # settings.num_workers = 40
            # If multi-class classification, the eval_metric 'auc' is removed from the list
            if class_number > 2:
                settings.eval_metric.remove('auc')
                settings.eval_metric.append('logloss')
                settings.eval_metric.append(F1ScoreMacro)
            # Flag to use embeddings in the tabnet model
            settings.use_embeddings = True
            # Threshold of the minimum of categorical features to use embeddings
            settings.threshold_categorical_features = 150
            # Use cat_emb_dim to set the embedding size for each categorical feature, for TabNet classifier
            # Flag to use cat_emb_dim to define the embedding size for each categorical feature, with False the embedding size is 1
            settings.use_cat_emb_dim = True
            # 'cpu' for cpu training, 'gpu' for gpu training, 'auto' to automatically detect gpu
            settings.device_name = 'cpu'
            # Apply custom data augmentation pipeline during training (parameter for fit method)
            # settings.augmentations = ClassificationSMOTE(
            #     p=0.2, device_name=settings.device_name)  # aug, None
            # 0 for no balancing, 1 for automated balancing, dict for custom weights per class, default 0 (parameter for fit method)
            # settings.weights = 0
            # Number of examples per batch. For larger dataset set 16384 (parameter for fit method)
            # settings.batch_size = 1024
            # Size of the mini batches used for "Ghost Batch Normalization". /!\ virtual_batch_size should divide batch_size. For larger dataset set 2048 (parameter for fit method)
            # settings.virtual_batch_size = 128

            # Show settings of the project
            reports.show_settings(settings=settings)

            # Delete duplicated rows by attribute
            precoce_ms_data_frame = pre_processing.delete_duplicate_rows_by_attribute(
                data_frame=precoce_ms_data_frame, attribute_name='ID_ANIMAL')

            # Delete column by names
            precoce_ms_data_frame = utils.delete_columns(
                data_frame=precoce_ms_data_frame, delete_columns_names=['ID_ANIMAL'])

            # if not is_batch_dataset:
            # Delete NaN rows
            precoce_ms_data_frame = pre_processing.delete_nan_rows(
                data_frame=precoce_ms_data_frame)

            # Convert pandas dtypes to numpy dtypes, some operations doesn't work with pandas dtype, for exemple, the XGBoost models
            precoce_ms_data_frame = utils.convert_pandas_dtype_to_numpy_dtype(
                data_frame=precoce_ms_data_frame, pandas_dtypes=[pd.UInt8Dtype()])

            # TODO: Maybe implement remove outliers. To detect outliers, use pre_processing.detect_outliers

            # Identify columns that contain a single value, and delete them
            precoce_ms_data_frame = pre_processing.delete_columns_with_single_value(
                data_frame=precoce_ms_data_frame
            )

            # # If binary classification, apply label encoder to the target column
            # if class_number == 2:
            # Apply label encoder to the columns
            precoce_ms_data_frame, settings.columns_label_encoded = pre_processing.label_encoder_columns(
                data_frame=precoce_ms_data_frame, columns_label_encoded=settings.columns_label_encoded,
                columns_names=settings.label_encoder_columns_names
            )

            # Save the label encoded columns
            utils.dump_joblib(
                object=settings.columns_label_encoded[settings.class_column],
                file_name='target_encoded',
                path_save_file=settings.PATH_SAVE_ENCODERS_SCALERS
            )

            # Move the target column to the last position in dataframe
            precoce_ms_data_frame = utils.move_cloumns_last_positions(
                data_frame=precoce_ms_data_frame, columns_names=[settings.class_column])

            # Target attribute distribution
            reports.class_distribution(
                y=precoce_ms_data_frame[settings.class_column].values)

            # Create x, the features, and y, the target
            x, y = utils.create_x_y_dataframe_data(
                data_frame=precoce_ms_data_frame
            )
            # # Either multiclass classification, apply label binarizer to the target column
            # else:
            #     # Apply label binarization to the class column
            #     x, y, settings.columns_label_binarized = pre_processing.label_binarizer_column(
            #         data_frame=precoce_ms_data_frame, columns_label_binarized=settings.columns_label_binarized,
            #         column_name=settings.class_column
            #     )

            # Create the fransformers for ColumnTransformer
            # transformers = list()
            # if is_batch_dataset:
            #     transformers = [
            #         pre_processing.create_simple_imputer_transformer(
            #             columns=settings.simple_imputer_columns_names,
            #             data_frame_columns=precoce_ms_data_frame.columns,
            #             strategy='most_frequent'
            #         ),
            #         pre_processing.create_ordinal_encoder_transformer(
            #             ordinal_encoder_columns_names=settings.ordinal_encoder_columns_names,
            #             data_frame_columns=precoce_ms_data_frame.columns,
            #         ),
            #         # pre_processing.create_ordinal_encoder_transformer(
            #         #     ordinal_encoder_columns_names=settings.ordinal_encoder_columns_names,
            #         #     data_frame_columns=precoce_ms_data_frame.columns,
            #         #     handle_unknown='use_encoded_value',
            #         #     unknown_value=-1,
            #         #     with_categories=False
            #         # ),
            #         pre_processing.create_one_hot_encoder_transformer(
            #             columns=settings.one_hot_encoder_columns_names,
            #             data_frame_columns=precoce_ms_data_frame.columns
            #         ),
            #         pre_processing.create_min_max_scaler_transformer(
            #             columns=settings.min_max_scaler_columns_names,
            #             data_frame_columns=precoce_ms_data_frame.columns,
            #             imputer=pre_processing.instance_simple_imputer(
            #                 strategy='mean')
            #         )
            #     ]
            # else:
            transformers = [
                pre_processing.create_ordinal_encoder_transformer(
                    ordinal_encoder_columns_names=settings.ordinal_encoder_columns_names,
                    data_frame_columns=precoce_ms_data_frame.columns,
                ),
                # pre_processing.create_ordinal_encoder_transformer(
                #     ordinal_encoder_columns_names=settings.ordinal_encoder_columns_names,
                #     data_frame_columns=precoce_ms_data_frame.columns,
                #     handle_unknown='use_encoded_value',
                #     unknown_value=-1,
                #     with_categories=False
                # ),
                pre_processing.create_one_hot_encoder_transformer(
                    columns=settings.one_hot_encoder_columns_names,
                    data_frame_columns=precoce_ms_data_frame.columns
                ),
                pre_processing.create_min_max_scaler_transformer(
                    columns=settings.min_max_scaler_columns_names,
                    data_frame_columns=precoce_ms_data_frame.columns
                )
            ]

            # Create the ColumnTransformer, for preprocessing the data in pipeline
            preprocessor = ColumnTransformer(
                transformers=transformers,
                verbose_feature_names_out=False,
                remainder='passthrough',
                n_jobs=-1
            )

            # Save the representation of the ColumnTransformer
            utils.save_estimator_repr(
                estimator=preprocessor,
                file_name='column_transformer',
                path_save_file=settings.PATH_SAVE_ESTIMATORS_REPR
            )

            # Test for the ColumnTransformer
            # preprocessor.fit(x_train)
            # print(preprocessor.get_feature_names_out())
            # print(preprocessor.transform(x_train))

            # Create the pipeline
            pipe = Pipeline(
                steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', ClfSwitcher())
                ]
            )

            # Save the representation of the pipeline
            utils.save_estimator_repr(
                estimator=pipe,
                file_name='pipeline',
                path_save_file=settings.PATH_SAVE_ESTIMATORS_REPR
            )

            # https://github.com/dreamquark-ai/tabnet/issues/288
            # The link above show the suggest parameters to use in GridSearchCV, and book name's where the parameters are explained
            param_grid = [
                {
                    'classifier__estimator': [GaussianNB()]
                },
                # {
                #     'classifier__estimator': [KNeighborsClassifier()],
                #     'classifier__estimator__n_jobs': [-1],
                #     'classifier__estimator__algorithm': ['kd_tree'],
                #     'classifier__estimator__metric': ['minkowski', 'euclidean'],
                #     'classifier__estimator__n_neighbors': list(np.arange(5, 17, 3)),
                #     'classifier__estimator__weights': ['uniform', 'distance'],
                #     'classifier__estimator__p': [1, 2, 3]
                # },
                {
                    'classifier__estimator': [DecisionTreeClassifier()],
                    'classifier__estimator__splitter': ['best'],
                    'classifier__estimator__random_state': [settings.random_seed],
                    'classifier__estimator__criterion': ['gini', 'entropy'],
                    'classifier__estimator__min_samples_split': [1, 2, 50, 100],
                    'classifier__estimator__min_samples_leaf': [1, 5, 10],
                    'classifier__estimator__max_depth': list(np.arange(1, 11, 3)) + [None],
                    'classifier__estimator__class_weight': ['balanced', None]
                },
                {
                    'classifier__estimator': [LinearSVC()],
                    'classifier__estimator__random_state': [settings.random_seed],
                    'classifier__estimator__dual': [False],
                    'classifier__estimator__penalty': ['l1', 'l2'],
                    'classifier__estimator__C': list(np.power(10, np.arange(-3, 1, dtype=np.float16))),
                    'classifier__estimator__max_iter': [100, 1000, 10000],
                    'classifier__estimator__class_weight': ['balanced', None]
                },
                {
                    'classifier__estimator': [MLPClassifier()],
                    'classifier__estimator__random_state': [settings.random_seed],
                    'classifier__estimator__max_iter': [1000],
                    'classifier__estimator__early_stopping': [True],
                    'classifier__estimator__hidden_layer_sizes': [(50, 100, 50), (100,), (200, 100)],
                    'classifier__estimator__activation': ['logistic', 'relu'],
                    'classifier__estimator__solver': ['adam', 'sgd'],
                    'classifier__estimator__alpha': [0.0001, 0.05],
                    'classifier__estimator__learning_rate': ['constant', 'adaptive'],
                    'classifier__estimator__learning_rate_init': [0.0001, 0.001],
                    'classifier__estimator__momentum': list(np.arange(0, 1, 0.3))
                },
                {
                    'classifier__estimator': [RandomForestClassifier()],
                    'classifier__estimator__random_state': [settings.random_seed],
                    'classifier__estimator__n_jobs': [4],
                    'classifier__estimator__criterion': ['entropy'],
                    'classifier__estimator__max_features': [0.75],
                    'classifier__estimator__n_estimators': [100, 1000],
                    'classifier__estimator__max_depth': [2, 7, None],
                    # 'classifier__estimator__min_samples_split': [2],
                    # 'classifier__estimator__min_samples_leaf': [1, 5, 10],
                    'classifier__estimator__class_weight': ['balanced', None]
                },
                {
                    # https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html#control-overfitting
                    # https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning/notebook
                    # https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
                    'classifier__estimator': [XGBClassifier()],
                    'classifier__estimator__tree_method': [settings.tree_method],
                    'classifier__estimator__max_delta_step': [1.0],
                    'classifier__estimator__random_state': [settings.random_seed],
                    'classifier__estimator__objective': [settings.objective],
                    'classifier__estimator__num_class': [class_number],
                    'classifier__estimator__n_jobs': [-1],
                    'classifier__estimator__subsample': [0.75],
                    'classifier__estimator__colsample_bytree': [0.75],
                    'classifier__estimator__n_estimators': [100, 1000],
                    'classifier__estimator__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__estimator__gamma': [0.05, 0.1, 1.0],
                    'classifier__estimator__max_depth': [2, 7, None],
                    # 'classifier__estimator__min_child_weight': [1, 7],
                    'classifier__estimator__reg_lambda': [0.01, 1.0],
                    'classifier__estimator__reg_alpha': [0, 1.0]
                },
                {
                    # https://www.kaggle.com/code/optimo/tabnetbaseline/notebook
                    # Using TabNetClassifier: https://github.com/dreamquark-ai/tabnet/issues/238
                    # https://github.com/dreamquark-ai/tabnet/blob/develop/census_example.ipynb
                    'classifier__estimator': [
                        TabNetClassifierTuner(
                            device_name=settings.device_name,
                            use_embeddings=settings.use_embeddings,
                            threshold_categorical_features=settings.threshold_categorical_features,
                            use_cat_emb_dim=settings.use_cat_emb_dim,
                            fit_eval_metric=settings.eval_metric,
                            fit_weights=settings.weights,
                            fit_batch_size=settings.batch_size,
                            fit_virtual_batch_size=settings.virtual_batch_size
                        )
                    ],
                    'classifier__estimator__seed': [settings.random_seed],
                    'classifier__estimator__clip_value': [1],
                    'classifier__estimator__verbose': [1],
                    'classifier__estimator__optimizer_fn': [torch.optim.Adam],
                    # 'classifier__estimator__optimizer_params': [dict(lr=2e-2)],
                    'classifier__estimator__optimizer_params': [
                        {'lr': 0.02},
                        {'lr': 0.01},
                        {'lr': 0.001}
                    ],
                    'classifier__estimator__scheduler_fn': [torch.optim.lr_scheduler.StepLR],
                    'classifier__estimator__scheduler_params': [{
                        'step_size': 10,  # how to use learning rate scheduler
                        'gamma': 0.95
                    }],
                    'classifier__estimator__mask_type': ['sparsemax'],
                    'classifier__estimator__n_a': [8, 64],
                    'classifier__estimator__n_steps': [3, 10],
                    'classifier__estimator__gamma': [1.3, 2.0],
                    'classifier__estimator__cat_emb_dim': [10, 20],
                    'classifier__estimator__n_independent': [2, 5],
                    'classifier__estimator__n_shared': [2, 5],
                    'classifier__estimator__momentum': [0.02, 0.4],
                    'classifier__estimator__lambda_sparse': [0.001, 0.1]
                }
            ]

            # Remove num_class parameter from XGBClassifier when using binary classification
            if class_number == 2:
                for estimator in param_grid:
                    if estimator['classifier__estimator'][0].__class__.__name__ == XGBClassifier().__class__.__name__:
                        estimator.pop('classifier__estimator__num_class')
                        break

            # Cross validation for grid search
            n_splits = 5
            print('Number of folds for cross validation: {}'.format(n_splits))
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=False
            )

            # TODO: Another way to search for the best parameters https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning/notebook
            # TODO: Indication of the advisor on how to load and save the parameters already executed in the grid search https://github.com/ragero/text_categorization_tool_python/blob/master/utilities/generate_parameters_list.py
            # TODO: Test this score: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
            # Custom refit strategy of a grid search with cross-validation (2 scores): https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py
            # Scoring strategy for grid search
            if class_number == 2:
                score = 'accuracy'
            else:
                score = 'f1_macro'
            print('Scoring strategy for grid search: {}'.format(score))

            # Size of test in train and test split
            # split_test_size = 0.2

            # Delete unused variables
            del precoce_ms_data_frame

            pattern_extraction.run_grid_search(
                x=x,
                y=y,
                estimator=pipe,
                param_grid=param_grid,
                cv=cv,
                score=score,
                n_jobs=settings.n_jobs,
                test_size=0.2,
                random_state=settings.random_seed,
                # error_score='raise'
            )

        ################################################## ANALYZE RESULTS #####################################

        if analyze_results:
            pass

        ################################################## PRE PROCESSING ##################################################

        if execute_pre_processing:

            # Delete duplicated rows by attribute
            precoce_ms_data_frame = pre_processing.delete_duplicate_rows_by_attribute(
                data_frame=precoce_ms_data_frame, attribute_name='ID_ANIMAL')

            # Delete column by names
            precoce_ms_data_frame = utils.delete_columns(
                data_frame=precoce_ms_data_frame, delete_columns_names=['ID_ANIMAL'])

            # Delete NaN rows
            precoce_ms_data_frame = pre_processing.delete_nan_rows(
                data_frame=precoce_ms_data_frame)

            # Convert pandas dtypes to numpy dtypes, some operations doesn't work with pandas dtype, for exemple, the XGBoost models
            precoce_ms_data_frame = utils.convert_pandas_dtype_to_numpy_dtype(
                data_frame=precoce_ms_data_frame, pandas_dtypes=[pd.UInt8Dtype()])

            # TODO: Maybe implement remove outliers. To detect outliers, use pre_processing.detect_outliers

            # Identify columns that contain a single value, and delete them
            precoce_ms_data_frame = pre_processing.delete_columns_with_single_value(
                data_frame=precoce_ms_data_frame
            )

            # Apply ordinal encoder to the columns
            precoce_ms_data_frame, settings.columns_ordinal_encoded = pre_processing.ordinal_encoder_columns(
                data_frame=precoce_ms_data_frame, columns_ordinal_encoded=settings.columns_ordinal_encoded,
                columns_names=settings.ordinal_encoder_columns_names
            )

            # Apply label encoder to the columns
            precoce_ms_data_frame, settings.columns_label_encoded = pre_processing.label_encoder_columns(
                data_frame=precoce_ms_data_frame, columns_label_encoded=settings.columns_label_encoded,
                columns_names=settings.label_encoder_columns_names
            )

            # Apply one hot encoder to the columns
            precoce_ms_data_frame, settings.columns_one_hot_encoded = pre_processing.one_hot_encoder_columns(
                data_frame=precoce_ms_data_frame, columns_one_hot_encoded=settings.columns_one_hot_encoded,
                columns_names=settings.one_hot_encoder_columns_names
            )

            # Apply min max scaler to the columns
            precoce_ms_data_frame, settings.columns_min_max_scaled = pre_processing.min_max_scaler_columns(
                data_frame=precoce_ms_data_frame, columns_min_max_scaled=settings.columns_min_max_scaled,
                columns_names=settings.min_max_scaler_columns_names
            )

            # Move the target column to the last position in dataframe
            precoce_ms_data_frame = utils.move_cloumns_last_positions(
                data_frame=precoce_ms_data_frame, columns_names=[settings.class_column])

            # Delete features by correlation
            precoce_ms_data_frame = pre_processing.drop_feature_by_correlation(
                data_frame=precoce_ms_data_frame, method='spearman', columns_names=settings.columns_names_drop_feature_by_correlation)

            # Target attribute distribution
            reports.class_distribution(
                y=precoce_ms_data_frame[settings.class_column].values)

            # TODO: Use function select_features_from_model implemented in the file pre_processing.py, in this location

            # TODO: Verify how to use sequential feature selector, and implement it

            # Apply inverse ordinal encoder to the columns
            precoce_ms_data_frame, settings.columns_ordinal_encoded = pre_processing.inverse_ordinal_encoder_columns(
                data_frame=precoce_ms_data_frame, columns_ordinal_encoded=settings.columns_ordinal_encoded)

            # Apply inverse label encoder to the columns
            precoce_ms_data_frame, settings.columns_label_encoded = pre_processing.inverse_label_encoder_columns(
                data_frame=precoce_ms_data_frame, columns_label_encoded=settings.columns_label_encoded)

            # Apply inverse one hot encoder to the columns
            precoce_ms_data_frame, settings.columns_one_hot_encoded = pre_processing.inverse_one_hot_encoder_columns(
                data_frame=precoce_ms_data_frame, columns_one_hot_encoded=settings.columns_one_hot_encoded)

            # Apply inverse min max scaler to the columns
            precoce_ms_data_frame, settings.columns_min_max_scaled = pre_processing.inverse_min_max_scaler_columns(
                data_frame=precoce_ms_data_frame, columns_min_max_scaled=settings.columns_min_max_scaled)

            # Move the target column to the last position in dataframe
            precoce_ms_data_frame = utils.move_cloumns_last_positions(
                data_frame=precoce_ms_data_frame, columns_names=[settings.class_column])

            ######################### NOT USED OR TESTED #########################
            reports.informations(precoce_ms_data_frame)

            path_save_csv_after_pre_processing = '/mnt/Dados/Mestrado_Computacao_Aplicada_UFMS/documentos_dissertacao/base_dados'

            csv_treatments.generate_new_csv(
                data_frame=precoce_ms_data_frame,
                csv_path=path_save_csv_after_pre_processing,
                csv_name='TAB_MODELAGEM_RAFAEL_2020_1_after_pre_processing-{}'.format(
                    utils.get_current_datetime()),
            )

        ################################################## PATTERN EXTRACTION ##################################################

        if execute_classifiers:

            x, y = utils.create_x_y_numpy_data(
                data_frame=precoce_ms_data_frame)

            print('\nX: ', type(x))
            print('Y: ', type(y))

            reports.class_distribution(y)

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
            settings.classifiers[KNeighborsClassifier().__class__.__name__] = [
                KNeighborsClassifier(), param_dist]

            # Naive bayes
            # Algorithm to be beaten. Used as base.
            # It's generating low values. Is it because the attributes are not discrete (words)?
            # For non-discrete attributes, should I use another algorithm?
            param_dist = {}
            settings.classifiers[GaussianNB().__class__.__name__] = [
                GaussianNB(), param_dist]

            # Decision Trees (c4.5)
            # Algorithm parameter settings, look for more (see documentation):
            # min_impurity_decrease ??? The algorithm presents an error when a value greater than 0.0 is added
            # ccp_alpha ??? the algorithm has an error when increasing the value
            # max_depth -> use a great search to calibrate
            # class_weight
            # I identified, that for the default parameters, the maximum depth of the created tree is 9. For the current amount of data.
            # criterion{“gini”, “entropy”},
            param_dist = {
                'criterion': ['gini', 'entropy'],
                'max_depth': list(np.arange(1, 11)) + [None],
                'random_state': [0],
                'class_weight': ['balanced', None]
            }
            settings.classifiers[DecisionTreeClassifier().__class__.__name__] = [
                DecisionTreeClassifier(), param_dist]

            # Neural Network
            # param_dist = {'solver': ['sgd'], 'learning_rate' : ['constant'], 'momentum' : scipy.stats.expon(scale=.1),
            # 'alpha' : scipy.stats.expon(scale=.0001), 'activation' : ['logistic'],
            # 'learning_rate_init' : scipy.stats.expon(scale=.01), 'hidden_layer_sizes':(200,100), 'max_iter':[500]}
            # learning_rate_init -> change this parameter if the result is not good
            # max_iter -> can also help to improve the result
            # hidden_layer_sizes -> (layer_x_with_y_neurons, layer_x_with_y_neurons)
            param_dist = {
                'solver': ['adam'],
                'learning_rate': ['constant'],
                'alpha': [0.001],
                'activation': ['relu'],
                'hidden_layer_sizes': (200, 100),
                'max_iter': [1000]
            }
            settings.classifiers[MLPClassifier().__class__.__name__] = [
                MLPClassifier(), param_dist]

            # Vector Support Machine
            # kernel: ‘linear’, ‘poly’, ‘rbf’
            #     # C: 10^x (-2 a 2)
            #     # 'max_iter': [100, 1000]
            #     # gamma : auto
            param_dist = {
                'kernel': ['linear', 'poly', 'rbf'],
                'C': list(np.power(10, np.arange(-2, 2, dtype=np.float16))),
                'max_iter': [10000],
                'gamma': ['auto']
            }
            settings.classifiers[SVC().__class__.__name__] = [
                SVC(), param_dist]

            # Random forest classifier
            param_dist = {
                'n_estimators': [10, 50, 100],
                'criterion': ['gini', 'entropy'],
                'max_depth': list(np.arange(1, 11)) + [None],
                'random_state': [0],
                'class_weight': ['balanced', None]
            }
            settings.classifiers[RandomForestClassifier().__class__.__name__] = [
                RandomForestClassifier(), param_dist]

            # Running the classifiers
            settings.models_results = pattern_extraction.run_models(
                x=x, y=y, models=settings.classifiers, models_results=settings.models_results)

            reports.models_results(
                models_results=settings.models_results, path_save_fig=settings.PATH_SAVE_PLOTS)

        tee_log_file.close()
    except Exception as e:
        print('Error: ', e)
        print(traceback.format_exc())
        tee_log_file.close()
        raise e
