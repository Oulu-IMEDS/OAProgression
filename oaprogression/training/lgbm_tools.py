import warnings
from functools import partial

import lightgbm as lgb
import numpy as np
import pandas as pd
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval
from tqdm import tqdm


def fit_lgb(params, train_folds, feature_set, metric, return_oof_res=False, return_models=False):
    oof_results = []
    clfs = []
    for fold_id, (train_split, val_split) in enumerate(train_folds):  # Going through the prepared fold splits
        d_train_prog = lgb.Dataset(train_split[feature_set], label=train_split.Progressor.values > 0)
        d_val_prog = lgb.Dataset(val_split[feature_set], label=val_split.Progressor.values > 0)
        with warnings.catch_warnings():
            # LGBM throws annoying messages, because we do not set the number
            # of iterations as a parameter
            warnings.simplefilter("ignore")
            clf_prog = lgb.train(params, d_train_prog, valid_sets=(d_train_prog, d_val_prog), verbose_eval=False)

        preds_prog = clf_prog.predict(val_split[feature_set], num_iteration=clf_prog.best_iteration)

        res = pd.DataFrame(data={'ID': val_split.ID.values, 'Side': val_split.Side.values, 'prog_pred': preds_prog,
                                 'Progressor': val_split.Progressor.values > 0})
        oof_results.append(res)
        clfs.append(clf_prog)

    oof_results = pd.concat(oof_results)
    res = list()
    res.append(metric(oof_results.Progressor.values.astype(int),
                      oof_results.prog_pred.values.astype(float)))

    if return_models:
        res.append(clfs)

    if return_oof_res:
        res.append(oof_results)

    if len(res) == 1:
        return res[0]
    else:
        return res


def init_lgbm_param_grid(seed):
    params = dict()
    params['num_iterations'] = hp.choice('num_iterations', [10, 100, 1000, 2000, 3000])
    params['early_stopping_round'] = hp.choice('early_stopping_round', [50, 100])
    params['learning_rate'] = hp.loguniform('learning_rate', -5, -3)
    params['boosting_type'] = hp.choice('boosting_type', ['gbdt', 'dart'])
    params['objective'] = 'binary'
    params['metric'] = 'binary_logloss'
    params['num_leaves'] = 2 + hp.randint('num_leaves', 21),
    params['max_depth'] = 3 + hp.randint('max_depth', 11),
    params['num_threads'] = 8
    params['feature_fraction'] = hp.uniform('feature_fraction', 0.6, 0.95)
    params['bagging_fraction'] = hp.uniform('bagging_fraction', 0.4, 0.95)
    params['bagging_freq'] = 1 + hp.randint('bagging_freq', 9),
    params['seed'] = seed
    params['bagging_seed'] = seed
    params['verbose'] = -1
    return params


def eval_lgb_objective(space, train_folds, feature_set, metric, callback=None):
    res = fit_lgb(space, train_folds, feature_set, metric, False, False)
    if callback is not None:
        callback()
    return {'loss': 1 - res, 'status': STATUS_OK}


def optimize_lgbm_hyperopt(train_folds, feature_set, metric, seed, hyperopt_trials=100):
    trials = Trials()
    pbar = tqdm(total=hyperopt_trials, desc="Hyperopt:")
    param_space = init_lgbm_param_grid(seed)
    best = fmin(fn=partial(eval_lgb_objective, train_folds=train_folds,
                           feature_set=feature_set, metric=metric, callback=lambda: pbar.update()),
                space=param_space,
                algo=tpe.suggest,
                max_evals=hyperopt_trials,
                trials=trials,
                verbose=0,
                rstate=np.random.RandomState(seed))
    pbar.close()
    return space_eval(param_space, best), trials
