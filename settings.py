import os

import utils

import pandas as pd
import torch

# Some settings are configured by default. If you want to change any settings,
# just follow the instruction for the specific setting.

# Set pandas max rows
pd.set_option('display.max_rows', utils.PANDAS_MAX_ROWS)

# Set random seed
random_seed = 42

# Set number of jobs to run in parallel
n_jobs = 1

# Size of test in train and test split
split_test_size = 0.2

############################################ CSV SETTINGS ############################################

# Folder path where the CSV file is located, i.e. '/Users/username/Desktop/'
dataset_folder_path = str

# Path to the dataset, i.e. '/Users/username/Desktop/dataset.csv'
csv_path = str

# Number of lines to be read from the dataset, where None read all lines
number_csv_lines = None

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
delete_columns_names_on_load_data = None

############################################ PATH SETTINGS ############################################

# Path to save plots
path_save_plots = './plots'

# Path to save composite estimators
path_save_estimators_repr = './estimators_repr'

############################################ ENCODERS SETTINGS ############################################

# Dictionary with column names to apply the ordinal encoder
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
# Dictionary with the ordinal encode object fitted for each column
columns_ordinal_encoded = dict()

# List with column names to apply the label encoder
label_encoder_columns_names = list()
# Dictionary with the label encoder object fitted for each column
columns_label_encoded = dict()

# List with column names to apply the ordinal encoder
one_hot_encoder_columns_names = list()
# Dictionary with the one hot encoder object fitted for each column
columns_one_hot_encoded = dict()

# List with column names to apply the min max scaler
min_max_scaler_columns_names = list()
# Dictionary with the min max scaler object fitted for each column
columns_min_max_scaled = dict()

############################################ CORRELATION SETTINGS ############################################

# List with column names to drop feature by correlation
columns_names_drop_feature_by_correlation = list()

############################################ TARGET SETTINGS ############################################

# Class column name
class_column = str

############################################ MODELS SETTINGS ############################################

# Dictionary containing the instantiated classes of classifiers and the parameters for optimization
classifiers = dict()

# Dictionary containing the execution results of the models
models_results = dict()

############################################ TABNET CLF TUNER SETTINGS ############################################

# 'cpu' for cpu training, 'gpu' for gpu training, 'auto' to automatically detect gpu
device_name = 'auto'

# Flag to use embeddings in the tabnet model, default True
use_embeddings = True

# List of categorical features indices
cat_idxs = list()

# List of categorical features number of modalities (number of unique values for a categorical feature) /!\ no new modalities can be predicted
cat_dims = list()

# Flag to use cat_emb_dim to define the embedding size for each categorical feature, with False the embedding size is 1
use_cat_emb_dim = False

# Embeddings size for each categorical features. (default =1)
cat_emb_dim = 1

# Threshold of the minimum of categorical features to use embeddings
threshold_categorical_features = 100

# Number of workers for the dataloader
num_workers = os.cpu_count() if torch.cuda.is_available() else 0

# List of evaluation metrics. The last metric is used for early stopping.
eval_metric = ['auc', 'accuracy', 'balanced_accuracy']

# Apply custom data augmentation pipeline during training, the default is None
augmentations = None

############################################  PERSISTENCE OBJECTS DURING RUN OF PIPELINE ############################################

# Flag to save the results of each split in the pipeline execution, to be used in a possible new execution, in case the execution is interrupted
save_results_during_run = True

# Whether True, the objects saved in the path_objects_persisted_results_will be cleaned before the execution of the pipeline
new_run = False

# Path to objects persisted with the results of executions of the pipeline
PATH_OBJECTS_PERSISTED_RESULTS_RUNS = './objects_persisted_results_runs'

# File name to save the parameters executed in the pipeline execution
PARAMETERS_PERSIST_FILENAME = utils.define_path_save_file(
    path_save_file=PATH_OBJECTS_PERSISTED_RESULTS_RUNS) + 'parameters_persist'

# File name to save all the results of each split of all estimator in the pipeline execution
RESULTS_PERSIST_FILENAME = utils.define_path_save_file(
    path_save_file=PATH_OBJECTS_PERSISTED_RESULTS_RUNS) + 'results_persist'

# File name to save the results of each split of estimator in the pipeline execution
SPLIT_PERSIST_FILENAME = utils.define_path_save_file(
    path_save_file=PATH_OBJECTS_PERSISTED_RESULTS_RUNS) + 'split_persist'
