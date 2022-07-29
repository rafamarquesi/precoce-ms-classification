from copy import deepcopy
import pandas as pd
import numpy as np
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split


def create_x_y_data(data_frame: pd.DataFrame) -> tuple:
    """Create x and y data from a DataFrame.

    Args:
        data_frame (pd.DataFrame): DataFrame with last cloumn as a targe class (y).

    Returns:
        tuple: Return x and y data from the DataFrame, being x and y of type numpy.array.
    """
    x = np.array(data_frame)[:, :-1]
    y = np.array(data_frame)[:, -1]
    return x, y


def run_models(x: np.array, y: np.array, models: dict, models_results: dict, n_splits: int = 10, shuffle: bool = True, random_state: int = 0) -> dict:
    """
    Run models and return the results of each model.

    Args:
        x (numpy.array): Data will be used to train the models.
        y (numpy.array): Real classes of the data.
        models (dict): Dictionary with the models to be used.
        models_results (dict): Dictionary with the results of the models.
        n_splits (int, optional): Number of splits of folds. Defaults to 10.
        shuffle (bool, optional): If True, shuffle the data. Defaults to True.
        random_state (int, optional): Random the class into the folds. Defaults to 0.

    Returns:
        dict: Return a dictionary with the avaliation results of each model.
    """

    if type_of_target(y) == 'unknown':
        y = np.array(y).astype(int)

    results = pd.DataFrame(
        columns=[
            'Iteração', 'Acurácia', 'Micro Revocação', 'Macro Revocação',
            'Micro Precisao', 'Macro Precisao', 'Micro F1', 'Macro F1'
        ]
    )

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )

    print('\n*****INICIO RUN MODELS******')

    count = 1

    for train_index, test_index in skf.split(X=x, y=y):
        # Split data into train and test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print('\n========== Iteração {} =========='.format(count))

        # Definning the best parameters for the models
        for key, models_params in models.items():
            model = __tunning_parameters(
                model_params=models_params, x=x_train, y=y_train)

            print('-------> Executando o Algoritmo: {}'.format(key))

            if key not in models_results:
                models_results[key] = results

            # Building the model
            model.fit(X=x_train, y=y_train)

            # Predicting the test data
            y_pred = model.predict(x_test)

            label_not_predicted = set(y_test) - set(y_pred)
            if len(label_not_predicted) > 0:
                print('Na iteração {}, não foi predito o Label: {}'.format(
                    count, label_not_predicted))

            # Calculating the metrics
            dict_results = __evaluate_model(y_test=y_test, y_pred=y_pred)
            dict_results['Iteração'] = count
            # models_results[key] = models_results[key].append(
            #     dict_results, ignore_index=True)
            models_results[key] = pd.concat(
                [models_results[key], pd.DataFrame.from_records([dict_results])], ignore_index=True)

        count += 1

    print('*****FIM RUN MODELS******')

    return models_results


############# PRIVATE METHODS #############


def __tunning_parameters(model_params: dict, x: np.array, y: np.array, train_size: float = 0.70, test_size: float = 0.30, n_splits: int = 3, shuffle: bool = True, random_state: int = 0) -> object:
    """
    Tunning the parameters of a model.

    Args:
        model_params (dict): Dictionary with the instantiated class of the classifier and the parameters for tuning (position 0 = instantiated class, position 1 = parameters for tuning)
        x (numpy.array): Data will be used to tunning models.
        y (numpy.array): Real class will be used to tunning models.
        train_size (float, optional): Percentage of data to be used to train the model. Defaults to 0.70.
        test_size (float, optional): Percentage of data to be used to test the model. Defaults to 0.30.
        n_splits (int, optional): Number of splits of folds. Defaults to 3.
        shuffle (bool, optional): If True, shuffle the data. Defaults to True.
        random_state (int, optional): Random the class into the folds. Defaults to 0.

    Returns:
        object: Return the instantiated model with the best parameters.
    """

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=train_size, test_size=test_size, shuffle=shuffle, random_state=random_state)

    model = deepcopy(model_params[0])
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )

    model_name = model.__class__.__name__
    best_params = None

    print('\n\n------ STARTED {} parameters tuning'.format(model_name))
    if model_params[1]:
        grid_search = GridSearchCV(
            model, param_grid=model_params[1], cv=skf, scoring='accuracy', refit=False
        )
        grid_search.fit(X=x_train, y=y_train)
        best_params = grid_search.best_params_
        model_params[0].set_params(**best_params)
        # model = model.set_params(**best_params)
    else:
        print('Params of model {} not defined. Keeping the default parameters.'.format(
            model_name))

    print('{} best params: {}'.format(model_name, best_params))
    print('------ FINISHED {} parameters tuning\n\n'.format(model_name))

    return model_params[0]


def __evaluate_model(y_test: np.array, y_pred: np.array) -> dict:
    """ Evaluate the model and return the metrics.

    Args:
        y_test (np.array): Real classes of the data.
        y_pred (np.array): Predicted classes of the data.

    Returns:
        dict: Return the metrics of the model.
    """

    dict_results = {}
    dict_results['Acurácia'] = accuracy_score(y_test, y_pred)
    dict_results['Micro Revocação'] = recall_score(
        y_test, y_pred, average='micro')
    dict_results['Macro Revocação'] = recall_score(
        y_test, y_pred, average='macro')
    dict_results['Micro Precisao'] = precision_score(
        y_test, y_pred, average='micro', labels=np.unique(y_pred))
    dict_results['Macro Precisao'] = precision_score(
        y_test, y_pred, average='macro', labels=np.unique(y_pred))
    dict_results['Micro F1'] = f1_score(y_test, y_pred, average='micro')
    dict_results['Macro F1'] = f1_score(y_test, y_pred, average='macro')

    return dict_results
