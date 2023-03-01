import os
from joblib import Parallel
from joblib import load
from collections import defaultdict
from itertools import product
import time
import warnings

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.utils.validation import indexable, _check_fit_params
from sklearn.model_selection._split import check_cv
from sklearn.base import is_classifier, clone
from sklearn.utils.fixes import delayed
from sklearn.model_selection._validation import _warn_or_raise_about_fit_failures
from sklearn.model_selection._validation import _insert_error_scores

from ._validation_tuner import _fit_and_score_tuner
import settings

__all__ = ['GridSearchCVTuner']


class GridSearchCVTuner(GridSearchCV):
    """Tuner for class GridSearchCV

    Args:
        GridSearchCV (class): Exhaustive search over specified parameter values for an estimator, from sklearn.model_selection
    """

    def fit(self, X, y=None, *, groups=None, **fit_params):
        """Custom function to keep the results of each cross-validation division, so if necessary, 
        if the execution is interrupted, the execution can be resumed without losing all the results already obtained.
        To perform the persistence of results, the joblib library is used.
        The added code is commented for better understanding of the change.

        Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).

        **fit_params : dict of str -> object
            Parameters passed to the `fit` method of the estimator.

            If a fit parameter is an array-like whose length is equal to
            `num_samples` then it will be split across CV groups along with `X`
            and `y`. For example, the :term:`sample_weight` parameter is split
            because `len(sample_weights) = len(X)`.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        estimator = self.estimator
        refit_metric = "score"

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit

        X, y, groups = indexable(X, y, groups)
        fit_params = _check_fit_params(X, fit_params)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
            # --------------------- START code to persist results ---------------------
            save_results_during_run=settings.save_results_during_run
            # --------------------- END code to persist results ---------------------
        )
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None, more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)

                # --------------------- START code to persist results ---------------------
                if not settings.new_run:

                    if os.path.exists(settings.PARAMETERS_PERSIST_FILENAME):
                        params_fitted = load(
                            settings.PARAMETERS_PERSIST_FILENAME)
                    else:
                        raise Exception('The file with the parameters already fitted does not exist. Please, check the path: {}'.format(
                            settings.PARAMETERS_PERSIST_FILENAME))

                    if os.path.exists(settings.RESULTS_PERSIST_FILENAME):
                        out_params_fitted = load(
                            settings.RESULTS_PERSIST_FILENAME)
                    else:
                        raise Exception('The file with the results already fitted does not exist. Please, check the path: {}'.format(
                            settings.RESULTS_PERSIST_FILENAME))

                    if n_splits != (len(out_params_fitted)/len(params_fitted)):
                        raise Exception('The number of folds configured in the GridSearchCV is different from the number of folds already executed.\nThe number of folds configured in the GridSearchCV is {}.\nThe number of folds in the file with the results already fitted is {}'.format(
                            n_splits, len(out_params_fitted)/len(params_fitted)))

                    if params_fitted and out_params_fitted:
                        print('Checking already executed parameters...')

                        remove_indexes = []

                        for i, params in enumerate(params_fitted):
                            remove = False
                            for key, value in params.items():
                                if key in candidate_params[i]:
                                    try:
                                        if candidate_params[i][key] == value:
                                            print('{} {} == {} {}'.format(
                                                key, value, key, candidate_params[i][key]))
                                            remove = True
                                        # Verify if the object is the same type of class and have same attributes
                                        elif (type(value) == type(candidate_params[i][key])) and (list(value.__dict__.keys()) == list(candidate_params[i][key].__dict__.keys())):
                                            print('{} {} == {} {}'.format(
                                                key, value, key, candidate_params[i][key]))
                                            remove = True
                                        else:
                                            print('{} {} != {} {}'.format(
                                                key, value, key, candidate_params[i][key]))
                                            remove = False
                                            break
                                    except AttributeError:
                                        print('{} {} != {} {}'.format(
                                            key, value, key, candidate_params[i][key]))
                                        remove = False
                                        break
                                else:
                                    print('{} {} != {} {}'.format(
                                        key, value, key, candidate_params[i][key]))
                                    remove = False
                                    break
                            if remove:
                                remove_indexes.append(i)

                            print(
                                '------------------------------------------------------------------')

                        for i in sorted(remove_indexes, reverse=True):
                            print('Removing already executed params object from candidate_params: {}\n'.format(
                                candidate_params[i]))
                            candidate_params.pop(i)

                        _warn_or_raise_about_fit_failures(
                            out_params_fitted, self.error_score)

                        # For callable self.scoring, the return type is only know after
                        # calling. If the return type is a dictionary, the error scores
                        # can now be inserted with the correct key. The type checking
                        # of out will be done in `_insert_error_scores`.
                        if callable(self.scoring):
                            _insert_error_scores(
                                out_params_fitted, self.error_score)

                        all_candidate_params.extend(params_fitted)
                        all_out.extend(out_params_fitted)
                    else:
                        warnings.warn(
                            'No fitted parameters were found, with their respective results, persisted on disk. Running all candidate_params.')
                # --------------------- END code to persist results ---------------------

                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        "Fitting {0} folds for each of {1} candidates,"
                        " totalling {2} fits".format(
                            n_splits, n_candidates, n_candidates * n_splits
                        )
                    )

                out = parallel(
                    delayed(_fit_and_score_tuner)(  # _fit_and_score_tuner is a custom function to persist results
                        clone(base_estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        parameters=parameters,
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        **fit_and_score_kwargs,
                    )
                    for (cand_idx, parameters), (split_idx, (train, test)) in product(
                        enumerate(candidate_params), enumerate(
                            cv.split(X, y, groups))
                    )
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        "inconsistent results. Expected {} "
                        "splits, got {}".format(
                            n_splits, len(out) // n_candidates)
                    )

                _warn_or_raise_about_fit_failures(out, self.error_score)

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]["test_scores"]
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = self._select_best_index(
                self.refit, refit_metric, results
            )
            if not callable(self.refit):
                # With a non-custom callable, we can select the best score
                # based on the best index
                self.best_score_ = results[f"mean_test_{refit_metric}"][
                    self.best_index_
                ]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(
                clone(base_estimator).set_params(**self.best_params_)
            )
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self
