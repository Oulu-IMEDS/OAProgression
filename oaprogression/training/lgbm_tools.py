import lightgbm as lgb
import pandas as pd

def fit_lgb(params, train_folds, feature_set, metric, return_oof_res=False, return_models=False):
    oof_results = []
    clfs = []
    for fold_id, (train_split, val_split) in enumerate(train_folds):  # Going through the prepared fold splits
        X_train = train_split[feature_set].values.astype(float)
        X_val = val_split[feature_set].values.astype(float)
        
        y_train = train_split.Progressor.values > 0
        y_val = val_split.Progressor.values > 0

        d_train_prog = lgb.Dataset(X_train.drop(features_drop, axis=1), label=y_train)
        d_val_prog = lgb.Dataset(X_val.drop(features_drop, axis=1), label=y_val)

        clf_prog = lgb.train(params, d_train_prog, 1000, valid_sets=(d_train_prog, d_val_prog), verbose_eval=False, \
                             early_stopping_rounds=50,evals_result=evals_result)

        preds_prog = clf_prog.predict(X_val.drop(features_drop, axis=1), num_iteration=clf_prog.best_iteration)

        res = pd.DataFrame(data={'ID': X_val.ID.values, 'Side': X_val.Side.values, 'prog_pred': preds_prog, 'Progressor': X_val.Progressor.values > 0 })
        oof_results.append(res)
        clfs.append(clfs)

    oof_results = pd.concat(oof_results)

    res = []
    res.append(metric(oof_results.Progressor, oof_results.prog_pred))
    
    if return_models:
        res.append(clfs)

    if return_oof_res:
        res.append(oof_results)

    if len(res) == 1:
        return res[0]
    else:
        return res


    
