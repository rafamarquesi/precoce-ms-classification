import warnings
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import settings
from pytorch_tabnet_tuner.utils_tuner import ComplexEncoderTuner


class TabNetClassifierTuner(TabNetClassifier):
    """A custom TabNetClassifier that can be tuned with a parameter search algorithm, for example: Grid Search."""

    def fit(self, X: np.ndarray, y: np.ndarray, *args, **kwargs):
        """Prepares X and y to be trained on the neural network, using a parameter search algorithm, for example: Grid Search

        When using embedding, the attributes cat_idxs, cat_dims, and, if selected to be used, cat_emb_dim, are created at runtime, after the split.

        If cat_emb_dim selected to be used, then a calculation is applied for its generation, otherwise, the value is 1.

        Args:
            X (np.ndarray): Train set
            y (np.ndarray): Train targets
        """

        # Dirty trick => would be better to add n_d in grid, or fix it in __init__ of tuner
        # self.n_d = self.n_a
        self.__update__(**{'n_d': self.n_a})
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=settings.random_seed, shuffle=True, stratify=y
        )

        if settings.use_embeddings:
            settings.cat_idxs = list()
            settings.cat_dims = list()
            settings.cat_emb_dim = 1

            for idxs, col in enumerate(X_train.T):
                unique_counts = len(np.unique(col))
                if unique_counts < settings.threshold_categorical_features:
                    settings.cat_idxs.append(idxs)
                    settings.cat_dims.append(unique_counts)

            if settings.use_cat_emb_dim:
                settings.cat_emb_dim = [
                    min(nb // 2, self.cat_emb_dim) for nb in settings.cat_dims
                ]
            elif self.cat_emb_dim != 1:
                warnings.warn(
                    'The variable cat_emb_dim has a value different from 1. If you want to use embeddings with cat_emb_dim, set the value of use_cat_emb_dim, in the settings, to True. Otherwise, the value of cat_emb_dim will be 1.')

            # -------- Attempt to solve problem: IndexError: index out of range in self
            # -------- It was found that X_train and X_valid have unique values that differ.
            # -------- I tried to remove the columns that have unique values that differ in X_train and X_valid.
            # idxs_del = list()
            # for idxs in range(0, X_train.shape[1]):
                # unique_counts_x_train = np.unique(
                #     X_train[:, idxs], return_counts=True)
                # unique_counts_x_valid = np.unique(
                #     X_valid[:, idxs], return_counts=True)
                # if len(unique_counts_x_train[0]) != len(unique_counts_x_valid[0]):
                #     idxs_del.append(idxs)

            # if idxs_del:
            #     X_train = np.delete(X_train, idxs_del, axis=1)
            #     X_valid = np.delete(X_valid, idxs_del, axis=1)

            # for idxs, col in enumerate(X_train.T):
            #     unique_counts = len(np.unique(col, return_counts=True)[1])
            #     if unique_counts < settings.threshold_categorical_features:
            #         settings.cat_idxs.append(idxs)
            #         settings.cat_dims.append(unique_counts)

            # if settings.use_cat_emb_dim:
            #     settings.cat_emb_dim = [
            #         min(nb // 2, settings.max_dim) for nb in settings.cat_dims
            #     ]

            # debug for the problem with split when has unique values differente in train and valid
            # for idxs in settings.cat_idxs:
            #     print('X idx:{} - unique:{}'.format(idxs, np.unique(X[:,idxs], return_counts=True)))
            #     print('X_train idx:{} - unique:{}'.format(idxs, np.unique(X_train[:,idxs], return_counts=True)))
            #     print('X_valid idx:{} - unique:{}\n'.format(idxs, np.unique(X_valid[:,idxs], return_counts=True)))
            #     print('Idx:{} - diffi X_train symmetric X_valid:{}\n\n'.format(idxs,( set(np.unique(X_train[:,idxs], return_counts=True)[0]).symmetric_difference(set(np.unique(X_valid[:,idxs], return_counts=True)[0])) )))

            self.__update__(
                **{
                    'cat_dims': settings.cat_dims,
                    'cat_emb_dim': settings.cat_emb_dim,
                    'cat_idxs': settings.cat_idxs
                }
            )

        super().fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_name=['train', 'valid'],
            # If needed implement F1-score metric: https://github.com/dreamquark-ai/tabnet/issues/245#issuecomment-739745307
            eval_metric=settings.eval_metric,
            weights=settings.weights,
            max_epochs=1000,
            patience=20,
            # batch_size=1024,
            # virtual_batch_size=128,
            batch_size=16384,
            virtual_batch_size=2048,
            num_workers=settings.num_workers,
            drop_last=False,
            augmentations=settings.augmentations
        )

    def save_model(self, path):
        """This function was custom for serealize all types of numpy arrays.
        Chenged ComplexEncoder to ComplexEncoderTuner.

        Saving TabNet model in two distinct files.

        Parameters
        ----------
        path : str
            Path of the model.

        Returns
        -------
        str
            input filepath with ".zip" appended

        """
        saved_params = {}
        init_params = {}
        for key, val in self.get_params().items():
            if isinstance(val, type):
                # Don't save torch specific params
                continue
            else:
                init_params[key] = val
        saved_params["init_params"] = init_params

        class_attrs = {
            "preds_mapper": self.preds_mapper
        }
        saved_params["class_attrs"] = class_attrs

        # Create folder
        Path(path).mkdir(parents=True, exist_ok=True)

        # Save models params
        with open(Path(path).joinpath("model_params.json"), "w", encoding="utf8") as f:
            json.dump(saved_params, f, cls=ComplexEncoderTuner)

        # Save state_dict
        torch.save(self.network.state_dict(),
                   Path(path).joinpath("network.pt"))
        shutil.make_archive(path, "zip", path)
        shutil.rmtree(path)
        print(f"Successfully saved model at {path}.zip")
        return f"{path}.zip"


class F1ScoreMacro(Metric):
    """F1 macro score metric for TabNet."""

    def __init__(self):
        self._name = 'f1_macro'
        self._maximize = True

    def __call__(self, y_true, y_score):
        return f1_score(y_true, np.argmax(y_score, axis=1), average='macro')
