import sys
import traceback

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import torch
from pytorch_tabnet.augmentations import ClassificationSMOTE

from clf_switcher import ClfSwitcher
from pytorch_tabnet_tuner.tab_model_tuner import TabNetClassifierTuner, F1ScoreMacro

import csv_treatments
import pre_processing
import reports
import pattern_extraction
import utils
import settings
from tee import Tee

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
        settings.n_jobs = 120

        # Folder path where the CSV file is located
        settings.dataset_folder_path = '/home/externo/rafaelrm/base_dados/'

        # Path to the dataset
        settings.csv_path = csv_treatments.choose_csv_path(
            sampling='100', folder_path=settings.dataset_folder_path)

        # Class column name
        settings.class_column = 'classificacao'

        # Checks if it is batch separated animals dataset
        # It was necessary to create to do some validations while loading the dataset, as it was changed from the original.
        is_batch_dataset = True if settings.csv_path.find(
            'ANIMAIS-POR-LOTE') != -1 else False

        # List with columns to delete when loading dataset
        settings.delete_columns_names_on_load_data = [
            'EstabelecimentoMunicipio', 'Maturidade', 'Acabamento', 'Peso',
            'DataAbate', 'ANO',
            'Frigorifico_ID', 'Frigorifico_CNPJ', 'Frigorifico_RazaoSocial', 'Municipio_Frigorifico',
            'EstabelecimentoIdentificador', 'Data_homol', 'Questionario_ID',
            'FERTIIRRIGACAO', 'CONCEN_VOLUM', 'CREEPFEEDING', 'FORN_ESTRAT_SILAGEM', 'PROTEICO', 'PROTEICO_ENERGETICO',
            'RACAO_BAL_CONS_INFERIOR', 'SAL_MINERAL', 'SALMINERAL_UREIA', 'RACAOO_BAL_CONSUMO_IG', 'GRAO_INTEIRO',
            'ALTO_CONCENTR_VOLUM', 'ALTO_CONCENTRADO', 'BPA',
            'area so confinamento', 'Lista Trace', 'Motivo', 'data_homol_select', 'dif_datas',
            'DataAbate_6m_ANT', 'data12m', 'data6m', 'data3m', 'data1m', 'data7d',
            'med7d_formITUinst', 'med7d_preR_soja', 'med7d_preR_milho', 'med7d_preR_boi',
            'med1m_formITUinst', 'med1m_preR_soja', 'med1m_preR_milho', 'med1m_preR_boi',
            'med3m_preR_soja',
            'med6m_preR_soja',
            'med12m_preR_soja',
            'cnt7d_CL_ITUinst', 'cnt1m_CL_ITUinst', 'cnt3m_CL_ITUinst', 'cnt6m_CL_ITUinst', 'cnt12m_CL_ITUinst',
            'tot7d_Chuva', 'med7d_TempInst', 'med7d_TempMin', 'med7d_UmidInst', 'med7d_formITUmax', 'med7d_NDVI', 'med7d_EVI',
            'tot1m_Chuva', 'med1m_TempInst', 'med1m_UmidInst', 'med1m_NDVI', 'med1m_EVI',
            'med3m_TempInst', 'med3m_UmidInst', 'med3m_formITUmax', 'med3m_EVI',
            'med6m_TempInst', 'med6m_UmidInst', 'med6m_EVI',
            'med12m_TempInst', 'med12m_TempMin', 'med12m_UmidInst', 'med12m_EVI',
            # columns above removed because they have 19.352582% of missing values
            'boa cobertura vegetal, com baixa', 'erosaoo laminar ou em sulco igua',
            # column above removed because it will not have the attribute at the time of performing the prediction and the target is derived from this attribute
            # 'classificacao'
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
            'tot3m_Chuva', 'med3m_formITUinst', 'med3m_NDVI', 'med3m_preR_milho', 'med3m_preR_boi',
            'tot6m_Chuva', 'med6m_formITUinst', 'med6m_NDVI', 'med6m_preR_milho', 'med6m_preR_boi',
            'tot12m_Chuva', 'med12m_formITUinst', 'med12m_NDVI', 'med12m_preR_milho', 'med12m_preR_boi'
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
            'med3m_formITUinst', 'med3m_preR_boi', 'med6m_preR_boi',
            settings.class_column
        ]

        execute_classifiers_pipeline = True

        ################################################## CSV TREATMENTS ##################################################

        if is_batch_dataset:
            settings.parse_dates = ['DataAbate']

        # Load the dataset
        precoce_ms_data_frame = csv_treatments.load_data(
            csv_path=settings.csv_path, delete_columns_names=settings.delete_columns_names_on_load_data,
            number_csv_lines=settings.number_csv_lines, dtype_dict=settings.dtype_dict, parse_dates=settings.parse_dates
        )

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
            # settings.tree_method = 'hist'
            # Specify the learning task and the corresponding learning objective. 'binary:logistic' is for binary classification.
            if class_number > 2:
                settings.objective = 'multi:softmax'

            ##### Tab Net Settings #####
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
            # settings.device_name = 'cpu'
            # Apply custom data augmentation pipeline during training (parameter for fit method)
            # settings.augmentations = ClassificationSMOTE(
            #     p=0.2, device_name=settings.device_name)  # aug, None
            # 0 for no balancing, 1 for automated balancing, dict for custom weights per class, default 0 (parameter for fit method)
            # settings.weights = 0
            # Number of examples per batch. For larger dataset set 16384 (parameter for fit method)
            settings.batch_size = 16384
            # Size of the mini batches used for "Ghost Batch Normalization". /!\ virtual_batch_size should divide batch_size. For larger dataset set 2048 (parameter for fit method)
            settings.virtual_batch_size = 2048

            # Show settings of the project
            reports.show_settings(settings=settings)

            # Delete duplicated rows by attribute
            precoce_ms_data_frame = pre_processing.delete_duplicate_rows_by_attribute(
                data_frame=precoce_ms_data_frame, attribute_name='ID_ANIMAL')

            # Delete column by names
            precoce_ms_data_frame = utils.delete_columns(
                data_frame=precoce_ms_data_frame, delete_columns_names=['ID_ANIMAL'])

            if not is_batch_dataset:
                # Delete NaN rows
                precoce_ms_data_frame = pre_processing.delete_nan_rows(
                    data_frame=precoce_ms_data_frame)

                # Convert pandas dtypes to numpy dtypes, some operations doesn't work with pandas dtype, for exemple, the XGBoost models
                precoce_ms_data_frame = utils.convert_pandas_dtype_to_numpy_dtype(
                    data_frame=precoce_ms_data_frame, pandas_dtypes=[pd.UInt8Dtype()])

            # Identify columns that contain a single value, and delete them
            precoce_ms_data_frame = pre_processing.delete_columns_with_single_value(
                data_frame=precoce_ms_data_frame
            )

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

            # Create the fransformers for ColumnTransformer
            transformers = list()
            if is_batch_dataset:
                transformers = [
                    pre_processing.create_simple_imputer_transformer(
                        columns=settings.simple_imputer_columns_names,
                        data_frame_columns=precoce_ms_data_frame.columns,
                        strategy='most_frequent'
                    ),
                    pre_processing.create_ordinal_encoder_transformer(
                        ordinal_encoder_columns_names=settings.ordinal_encoder_columns_names,
                        data_frame_columns=precoce_ms_data_frame.columns,
                    ),
                    pre_processing.create_one_hot_encoder_transformer(
                        columns=settings.one_hot_encoder_columns_names,
                        data_frame_columns=precoce_ms_data_frame.columns
                    ),
                    pre_processing.create_min_max_scaler_transformer(
                        columns=settings.min_max_scaler_columns_names,
                        data_frame_columns=precoce_ms_data_frame.columns,
                        imputer=pre_processing.instance_simple_imputer(
                            strategy='mean')
                    )
                ]
            else:
                transformers = [
                    pre_processing.create_ordinal_encoder_transformer(
                        ordinal_encoder_columns_names=settings.ordinal_encoder_columns_names,
                        data_frame_columns=precoce_ms_data_frame.columns,
                    ),
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
                    # 'classifier__estimator__random_state': [settings.random_seed],
                    'classifier__estimator__max_iter': [1000],
                    'classifier__estimator__early_stopping': [True],
                    'classifier__estimator__hidden_layer_sizes': [(50, 100, 50), (100,), (200, 100)],
                    'classifier__estimator__activation': ['logistic', 'relu'],
                    'classifier__estimator__solver': ['adam', 'sgd'],
                    'classifier__estimator__alpha': [0.0001, 0.05],
                    'classifier__estimator__learning_rate': ['constant', 'adaptive'],
                    'classifier__estimator__learning_rate_init': [0.0001, 0.001],
                    'classifier__estimator__momentum': list(np.arange(0, 1, 0.3))
                }
            ]

            # Remove num_class parameter from XGBClassifier when using binary classification
            if class_number == 2:
                for estimator in param_grid:
                    if estimator['classifier__estimator'][0].__class__.__name__ == XGBClassifier().__class__.__name__:
                        estimator.pop('classifier__estimator__num_class')
                        break

            # Cross validation for grid search
            n_splits = 10
            print('Number of folds for cross validation: {}'.format(n_splits))
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=False
            )

            # Scoring strategy for grid search
            if class_number == 2:
                score = 'accuracy'
            else:
                score = 'f1_macro'
            print('Scoring strategy for grid search: {}'.format(score))

            # Delete unused variables
            del precoce_ms_data_frame

            # pattern_extraction.run_grid_search(
            #    x=x,
            #    y=y,
            #    estimator=pipe,
            #    param_grid=param_grid,
            #    cv=cv,
            #    score=score,
            #    n_jobs=settings.n_jobs,
            #    test_size=0.2,
            #    random_state=settings.random_seed,
            #    pre_dispatch=59,
            #    execution_name='GS1'
            # )

            settings.n_jobs = 15
            print('\n-------Number of jobs for grid search 2: {}'.format(settings.n_jobs))

            param_grid_gs2 = [
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
                }
            ]

            # Configuration run RandomForest
            #pattern_extraction.run_grid_search(
            #    x=x,
            #    y=y,
            #    estimator=pipe,
            #    param_grid=param_grid_gs2,
            #    cv=cv,
            #    score=score,
            #    n_jobs=settings.n_jobs,
            #    test_size=0.2,
            #    random_state=settings.random_seed,
            #    execution_name='GS2'
            #)

            settings.n_jobs = 10
            print('\n-------Number of jobs for grid search 3: {}'.format(settings.n_jobs))

            param_grid_gs3 = [
                {
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
                }
            ]

            # Configuration run XGBoost
            pattern_extraction.run_grid_search(
                x=x,
                y=y,
                estimator=pipe,
                param_grid=param_grid_gs3,
                cv=cv,
                score=score,
                n_jobs=settings.n_jobs,
                test_size=0.2,
                random_state=settings.random_seed,
                execution_name='GS3',
                error_score='raise'
            )

            settings.n_jobs = 20
            print('\n-------Number of jobs for grid search 4: {}'.format(settings.n_jobs))

            param_grid_gs4 = [
                {
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

            # Configuration run TabNet
            pattern_extraction.run_grid_search(
                x=x,
                y=y,
                estimator=pipe,
                param_grid=param_grid_gs4,
                cv=cv,
                score=score,
                n_jobs=settings.n_jobs,
                test_size=0.2,
                random_state=settings.random_seed,
                execution_name='GS4'
            )

        tee_log_file.close()
    except Exception as e:
        print('Error: ', e)
        print(traceback.format_exc())
        tee_log_file.close()
        raise e
