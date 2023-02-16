import utils
from sys import displayhook

from typing import Union
from copy import deepcopy

import pandas as pd
import numpy as np

from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, roc_auc_score, classification_report, cohen_kappa_score, matthews_corrcoef, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

import settings
import csv_treatments
import utils
import reports
from sklearn_tuner.model_selection_tuner import GridSearchCVTuner


@utils.timeit
def run_grid_search(
    x: pd.DataFrame,
    y: pd.Series,
    estimator: object,
    param_grid: Union[dict, list],
    cv: Union[None, int, object, iter] = None,
    score: str = 'accuracy',
    n_jobs: int = None,
    test_size: Union[float, int, None] = None,
    random_state: Union[int, np.random.RandomState, None] = None,
    verbose: int = 10,
    error_score=np.nan
) -> dict:
    """
    Run Grid Search CV and save the results, best parameters, and best model.

    Args:
        x (pd.DataFrame): Data will be used to train the models.
        y (pd.Series): Real classes of the data.
        estimator (object): Estimator to be used.
        param_grid (Union[dict, list]): Dictionary or list with the dictionaries with parameters to be tested.
        cv (Union[None, int, object, iter], optional): Cross-validation splitting strategy. Defaults to None.
        score (str, optional): Scoring metric. Defaults to 'accuracy'.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to None.
        test_size (Union[float, int, None], optional): Size of the test data. Defaults to None.
        random_state (int, optional): Random the class into the folds. Defaults to None.
        verbose (int, optional): Controls the verbosity: the higher, the more messages. Defaults to 10.
        error_score (optional): Value to assign to the score if an error occurs in estimator fitting. If set to 'raise', the error is raised. If a numeric value is given, FitFailedWarning is raised. This parameter does not affect the refit step, which will always raise the error. Defaults to np.nan.
    """

    # Split the data into test and train
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print('Test Size: {}\n'.format(test_size))
    print('x_train shape: {}'.format(x_train.shape))
    print('y_train shape: {}'.format(y_train.shape))
    print('x_test shape: {}'.format(x_test.shape))
    print('y_test shape: {}'.format(y_test.shape))

    if settings.save_results_during_run:

        if settings.new_run:
            utils.remove_all_files_in_directory(
                path_directory=settings.PATH_OBJECTS_PERSISTED_RESULTS_RUNS)

        grid_search = GridSearchCVTuner(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring=score,
            n_jobs=n_jobs,
            verbose=verbose,
            error_score=error_score
        )
    else:
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring=score,
            n_jobs=n_jobs,
            verbose=verbose,
            error_score=error_score
        )

    # Save the representation of the GridSearchCV
    utils.save_estimator_repr(
        estimator=grid_search,
        file_name='grid_search',
        path_save_file=settings.PATH_SAVE_ESTIMATORS_REPR
    )

    grid_search.fit(x_train, y_train)

    print('--------------------- RESULTS ---------------------')

    # Cross validation results in a DataFrame
    cv_results = pd.DataFrame.from_dict(
        grid_search.cv_results_, orient='columns')
    cv_results = cv_results.sort_values(
        'mean_test_score', ascending=False)
    csv_treatments.generate_new_csv(
        data_frame=cv_results,
        csv_path=settings.PATH_SAVE_RESULTS,
        csv_name='cv_results-{}'.format(utils.get_current_datetime())
    )
    print('Cross validation results:')
    displayhook(cv_results)

    # Stores the optimum model in best_pipe
    best_pipe = grid_search.best_estimator_

    # Save best pipe
    utils.dump_joblib(
        object=best_pipe,
        file_name='best_pipe',
        path_save_file=settings.PATH_SAVE_BEST_ESTIMATORS
    )

    # Save the representation of the best pipe in grid search
    utils.save_estimator_repr(
        estimator=best_pipe,
        file_name='best_pipe',
        path_save_file=settings.PATH_SAVE_ESTIMATORS_REPR
    )

    # Store the best model in best_estimator
    best_estimator = best_pipe.steps[-1][-1].estimator

    # Save best estimator
    utils.save_best_estimator(best_estimator=best_estimator)
    # utils.dump_joblib(
    #     object=best_estimator,
    #     file_name='best_estimator',
    #     path_save_file=settings.PATH_SAVE_BEST_ESTIMATORS
    # )

    # Save the representation of the best estimator in grid search
    utils.save_estimator_repr(
        estimator=best_estimator,
        file_name='best_estimator',
        path_save_file=settings.PATH_SAVE_ESTIMATORS_REPR
    )

    print('Best estimator: {}'.format(best_estimator))

    # Save the column transformer, for preprocessing data in prediction
    utils.dump_joblib(
        object=best_pipe.named_steps['preprocessor'],
        file_name='column_transformer',
        path_save_file=settings.PATH_SAVE_ENCODERS_SCALERS
    )

    # Dict with the grid search results
    grid_search_results = dict()

    print('Internal CV score obtained by the best set of parameters: {}'.format(
        grid_search.best_score_))
    # Save the best_score_ in grid_search_results
    grid_search_results['best_score_'] = grid_search.best_score_

    # Access the best set of parameters
    print('Best params: {}'.format(grid_search.best_params_))
    # Save the best_params_ in grid_search_results
    grid_search_results['best_params_'] = [grid_search.best_params_]

    # Scorer function used on the held out data to choose the best parameters for the model
    print('Scorer function: {}'.format(grid_search.scorer_))
    grid_search_results['scorer_'] = grid_search.scorer_

    # The number of cross-validation splits (folds/iterations)
    print('The number of CV splits: {}'.format(grid_search.n_splits_))
    grid_search_results['n_splits_'] = grid_search.n_splits_

    print('Seconds used for refitting the best model on the whole dataset: {}'.format(
        grid_search.refit_time_))
    grid_search_results['refit_time_'] = grid_search.refit_time_

    print('Whether the scorers compute several metrics: {}'.format(
        grid_search.multimetric_))
    grid_search_results['multimetric_'] = grid_search.multimetric_

    # Don't work!!!!
    # print('The classes labels: {}'.format(grid_search.classes_))
    # grid_search_results['classes_'] = grid_search.classes_

    print('The number of features when fit is performed: {}'.format(
        grid_search.n_features_in_))
    grid_search_results['n_features_in_'] = grid_search.n_features_in_

    print('Names of features seen during fit: {}'.format(
        grid_search.feature_names_in_))
    grid_search_results['feature_names_in_'] = [
        grid_search.feature_names_in_]

    # https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html#control-overfitting
    print('\n!!!>> When you observe high training accuracy, but low test accuracy, it is likely that you encountered overfitting problem.')

    training_set_score = grid_search.score(x_train, y_train)
    print('Training set score: {}'.format(training_set_score))
    grid_search_results['training_set_score'] = training_set_score

    test_set_score = grid_search.score(x_test, y_test)
    print('Test set score: {}'.format(test_set_score))
    grid_search_results['test_set_score'] = test_set_score

    csv_treatments.generate_new_csv(
        data_frame=pd.DataFrame.from_dict(
            grid_search_results, orient='columns'),
        csv_path=settings.PATH_SAVE_RESULTS,
        csv_name='grid_search_results-{}'.format(
            utils.get_current_datetime())
    )

    # Predict the test set
    y_pred = grid_search.predict(x_test)
    try:
        y_pred_proba = grid_search.predict_proba(x_test)
    except:
        y_pred_proba = None

    # Save confusion matrix
    reports.confusion_matrix_display(
        y_true=y_test,
        y_pred=y_pred,
        display_figure=False,
        path_save_fig=settings.PATH_SAVE_PLOTS
    )

    print('\n--- Test data performance ---')

    # Get the number of classes in the target column
    class_number = len(y.value_counts())

    dict_results = dict()

    dict_results['Acurácia'] = accuracy_score(y_test, y_pred)
    dict_results['Acurácia Balanceada'] = balanced_accuracy_score(
        y_test, y_pred)
    if class_number == 2:
        dict_results['Revocação'] = recall_score(y_test, y_pred)
    else:
        dict_results['Revocação Ponderada'] = recall_score(
            y_test, y_pred, average='weighted')
    dict_results['Micro Revocação'] = recall_score(
        y_test, y_pred, average='micro')
    dict_results['Macro Revocação'] = recall_score(
        y_test, y_pred, average='macro')
    if class_number == 2:
        dict_results['Precisão'] = precision_score(y_test, y_pred)
    else:
        dict_results['Precisão Ponderada'] = precision_score(
            y_test, y_pred, average='weighted')
    dict_results['Micro Precisão'] = precision_score(
        y_test, y_pred, average='micro', labels=np.unique(y_pred))
    dict_results['Macro Precisão'] = precision_score(
        y_test, y_pred, average='macro', labels=np.unique(y_pred))
    if class_number == 2:
        dict_results['F1'] = f1_score(y_test, y_pred)
    else:
        dict_results['F1 Ponderado'] = f1_score(
            y_test, y_pred, average='weighted')
    dict_results['Micro F1'] = f1_score(
        y_test, y_pred, average='micro')
    dict_results['Macro F1'] = f1_score(
        y_test, y_pred, average='macro')
    dict_results['Coeficiente Kappa'] = cohen_kappa_score(y_test, y_pred)
    dict_results['Coeficiente de Correlação de Matthews'] = matthews_corrcoef(
        y_test, y_pred)
    if y_pred_proba is not None:
        dict_results['Log Loss'] = log_loss(y_test, y_pred_proba)
        if class_number == 2:
            dict_results['ROC AUC Score'] = roc_auc_score(
                y_test, y_pred_proba[:, 1])
        else:
            dict_results['ROC AUC Score Ponderado'] = roc_auc_score(
                y_test, y_pred_proba, multi_class='ovr', average='weighted')

    for key, value in dict_results.items():
        print('Test {}: {}'.format(key, value))

    csv_treatments.generate_new_csv(
        data_frame=pd.DataFrame(
            dict_results, index=[0]),
        csv_path=settings.PATH_SAVE_RESULTS,
        csv_name='performance_results-{}'.format(
            utils.get_current_datetime())
    )

    # classification report : https://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection_pipeline.html#sphx-glr-auto-examples-feature-selection-plot-feature-selection-pipeline-py
    classification_report_str = classification_report(y_test, y_pred)
    print(classification_report_str)
    csv_treatments.generate_new_csv(
        data_frame=pd.DataFrame.from_records(
            [
                {
                    'classification_report_str': [classification_report_str],
                    'classification_report_dict': [classification_report(y_test, y_pred, output_dict=True)]
                }
            ]),
        csv_path=settings.PATH_SAVE_RESULTS,
        csv_name='classification_report-{}'.format(
            utils.get_current_datetime())
    )


@utils.timeit
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
            'Micro Precisao', 'Macro Precisao', 'Micro F1', 'Macro F1', 'Melhores Parâmetros'
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
            dict_results['Melhores Parâmetros'] = model.get_params()
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
