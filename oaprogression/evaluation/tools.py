import glob
import os
from functools import partial

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import solt.core as slc
import solt.transforms as slt
import statsmodels.api as sm
import torch
import torch.nn.functional as F
import torchvision.transforms as tv_transforms
from sklearn.metrics import roc_auc_score, cohen_kappa_score, confusion_matrix, mean_squared_error, f1_score, \
    average_precision_score
import gc

from tqdm import tqdm
from sklearn.metrics import roc_curve, precision_recall_curve
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from oaprogression.evaluation import stats
from oaprogression.training import model
from oaprogression.training import session as session
from oaprogression.training.dataset import OAProgressionDataset, unpack_solt_data, img_labels2solt, apply_by_index
from oaprogression.evaluation import gcam

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def five_crop(img, size):
    """Returns a stacked 5 crop
    """
    img = img.clone()
    h, w = img.size()[-2:]
    # get central crop
    c_cr = img[:, h // 2 - size // 2:h // 2 + size // 2, w // 2 - size // 2:w // 2 + size // 2]
    # upper-left crop
    ul_cr = img[:, 0:size, 0:size]
    # upper-right crop
    ur_cr = img[:, 0:size, w - size:w]
    # bottom-left crop
    bl_cr = img[:, h - size:h, 0:size]
    # bottom-right crop
    br_cr = img[:, h - size:h, w - size:w]
    return torch.stack((c_cr, ul_cr, ur_cr, bl_cr, br_cr))


def init_fold(fold_id, session_snapshot, fold_path, return_fc_kl=False):
    net = model.KneeNet(session_snapshot['args'][0].backbone, 0.5, False)
    snapshot_name = glob.glob(os.path.join(fold_path, f'fold_{fold_id}*.pth'))[0]

    net.load_state_dict(torch.load(snapshot_name))

    features = nn.DataParallel(net.features[:-1])
    fc = nn.DataParallel(net.classifier_prog[-1])

    features.to('cuda')
    fc.to('cuda')

    features.eval()
    fc.eval()

    if return_fc_kl:
        fc_kl = nn.DataParallel(net.classifier_kl[-1])
        fc_kl.to('cuda')
        fc_kl.eval()

        return features, fc, fc_kl

    return features, fc


def init_loader(metadata, args, snapshots_root):
    mean_vector, std_vector = session.init_mean_std(snapshots_root, None, None, None)

    norm_trf = tv_transforms.Normalize(mean_vector.tolist(), std_vector.tolist())

    tta_trf = tv_transforms.Compose([
        img_labels2solt,
        slc.Stream([
            slt.PadTransform(pad_to=(700, 700), padding='z'),
            slt.CropTransform(crop_size=(700, 700), crop_mode='c'),
            slt.ResizeTransform(resize_to=(310, 310), interpolation='bicubic'),
            slt.ImageColorTransform(mode='gs2rgb'),
        ], interpolation='bicubic'),
        unpack_solt_data,
        partial(apply_by_index, transform=tv_transforms.ToTensor(), idx=0),
        partial(apply_by_index, transform=norm_trf, idx=0),
        partial(apply_by_index, transform=partial(five_crop, size=300), idx=0),
    ])

    dataset = OAProgressionDataset(dataset=args.dataset_root,
                                   split=metadata, trf=tta_trf)

    loader = DataLoader(dataset,
                        batch_size=args.bs,
                        sampler=SequentialSampler(dataset),
                        num_workers=args.n_threads)

    return loader


def eval_batch(sample, features, fc_prog, fc_kl=None):
    with torch.no_grad():
        inputs = sample['img'].to("cuda")
        bs, ncrops, c, h, w = inputs.size()
        maps = features(inputs.view(-1, c, h, w))
        maps_avg = F.adaptive_avg_pool2d(maps, 1).view(maps.size(0), -1)
        out_prog = F.softmax(fc_prog(maps_avg), 1).view(bs, ncrops, -1).mean(1)

        if fc_kl is not None:
            out_kl = F.softmax(fc_kl(maps_avg), 1).view(bs, ncrops, -1).mean(1)
            return out_prog.to("cpu").numpy(), out_kl.to("cpu").numpy()

        return out_prog.to("cpu").numpy()


def run_test_inference(loader, session_snapshot, snapshots_root, snapshot, save_dir):
    gradcam_maps_all = 0
    res_kl = 0
    res_prog = 0
    ids = None
    for fold_id in range(session_snapshot['args'][0].n_folds):
        features, fc, fc_kl = init_fold(fold_id, session_snapshot, os.path.join(snapshots_root, snapshot),
                                        return_fc_kl=True)

        preds_prog_fold = []
        preds_kl_fold = []
        gradcam_maps_fold = []
        ids = []
        for batch_id, sample in enumerate(
                tqdm(loader, total=len(loader), desc='Prediction from fold {}'.format(fold_id))):
            gcam_batch, probs_prog, probs_kl = gcam.eval_batch(sample, features, fc, fc_kl)
            gradcam_maps_fold.append(gcam_batch)
            preds_prog_fold.append(probs_prog)
            preds_kl_fold.append(probs_kl)
            ids.extend(sample['ID_SIDE'])
            gc.collect()

        preds_prog_fold = np.vstack(preds_prog_fold)
        preds_kl_fold = np.vstack(preds_kl_fold)
        gradcam_maps_all += np.vstack(gradcam_maps_fold)

        res_kl += preds_kl_fold
        res_prog += preds_prog_fold
        gc.collect()

    res_kl /= 5.
    res_prog /= 5.
    np.savez_compressed(os.path.join(save_dir, 'results.npz'),
                        gradcam_maps_all=gradcam_maps_all,
                        preds_kl=res_kl,
                        preds_prog=res_prog,
                        ids=ids)


def calc_metrics(gt_progression, gt_kl, preds_progression, preds_kl):
    # Computing Validation metrics
    preds_progression_bin = preds_progression[:, 1:].sum(1)
    preds_kl_bin = preds_kl[:, 1:].sum(1)

    res = dict()
    res['cm_prog'] = confusion_matrix(gt_progression, preds_progression.argmax(1))
    res['cm_kl'] = confusion_matrix(gt_kl, preds_kl.argmax(1))
    res['auc_prog'] = roc_auc_score(gt_progression > 0, preds_progression_bin)
    res['kappa_prog'] = cohen_kappa_score(gt_progression, preds_progression.argmax(1), weights="quadratic")
    res['acc_prog'] = np.mean(res['cm_prog'].diagonal().astype(float) / res['cm_prog'].sum(axis=1))
    res['mse_prog'] = mean_squared_error(gt_progression, preds_progression.argmax(1))
    res['auc_oa'] = roc_auc_score(gt_kl > 1, preds_kl_bin)
    res['kappa_kl'] = cohen_kappa_score(gt_kl, preds_kl.argmax(1), weights="quadratic")
    res['acc_kl'] = np.mean(res['cm_kl'].diagonal().astype(float) / res['cm_kl'].sum(axis=1))
    res['mse_kl'] = mean_squared_error(gt_kl, preds_kl.argmax(1))
    res['f1_score_03_prog'] = f1_score(gt_progression > 0, preds_progression_bin > 0.3)
    res['f1_score_04_prog'] = f1_score(gt_progression > 0, preds_progression_bin > 0.5)
    res['f1_score_05_prog'] = f1_score(gt_progression > 0, preds_progression_bin > 0.6)
    res['ap_prog'] = average_precision_score(gt_progression > 0, preds_progression_bin)

    return res


def eval_models(metadata_test, feature_set, models_best,
                mean_std_best=None,
                impute=True,
                model_type='sklearn'):
    x_test_initial = metadata_test[feature_set].copy()
    # Using mean imputation if necessary
    if impute:
        x_test_initial.fillna(x_test_initial.mean(), inplace=True)
        x_test_initial = x_test_initial.values.astype(float)

    test_res = 0
    for model_id in range(len(models_best)):
        x_test = x_test_initial.copy()
        if mean_std_best is not None and model_type != 'lgbm':
            mean, std = mean_std_best[model_id]
            x_test -= mean
            x_test /= std
        clf_prog = models_best[model_id]
        if model_type == 'sklearn':
            test_res += clf_prog.predict_proba(x_test)[:, 1]
        elif model_type == 'lgbm':
            test_res += clf_prog.predict(x_test, clf_prog.best_iteration)
        elif model_type == 'statsmodels':
            test_res += clf_prog.predict(sm.add_constant(x_test)).flatten()
        else:
            raise ValueError

    test_res /= len(models_best)
    return test_res


def pkl2df(fname):
    data_dict = pd.read_pickle(fname)
    res = {}
    for key in data_dict:
        res[key] = pd.DataFrame(data={'ID': data_dict[key][0],
                                      'Side': data_dict[key][1],
                                      'Progressor': data_dict[key][2],
                                      'Prediction': data_dict[key][3]})
    return res


def init_auc_pr_plot(y):
    plt.rcParams.update({'font.size': 13})
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    axs[0].plot([0, 1], [0, 1], '--', color='black')
    axs[0].set_xlim([0, 1])
    axs[0].set_ylim([0, 1])
    axs[0].grid()
    axs[0].set_xlabel('False positive rate')
    axs[0].set_ylabel('True positive rate')
    axs[0].set_title('ROC curve')

    axs[1].axhline(y=y.sum() / y.shape[0], linestyle='--', color='black')
    axs[1].set_xlim([0, 1])
    axs[1].set_ylim([0, 1])
    axs[1].grid()
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].set_title('Precision-Recall curve')

    return fig, axs


def compute_and_plot_curves(tmp_df, axs, key=None, legend=True, color=None, n_bootstrap=2000, seed=12345):
    auc, ci_l, ci_h, fpr, tpr = stats.calc_curve_bootstrap(roc_curve, roc_auc_score,
                                                           tmp_df.Progressor.values.astype(int),
                                                           tmp_df.Prediction.values.astype(float),
                                                           n_bootstrap, seed, stratified=True, alpha=95)

    if key is None:
        key = ''
    if color is None:
        axs[0].plot(fpr, tpr, label=key + f' ({np.round(auc, 2)} [{np.round(ci_l, 2)}, {np.round(ci_h, 2)}])')
    else:
        axs[0].plot(fpr, tpr, label=key + f' ({np.round(auc, 2)} [{np.round(ci_l, 2)}, {np.round(ci_h, 2)}])',
                    color=color)
    if legend:
        axs[0].legend()

    ap, ci_l, ci_h, precision, recall = stats.calc_curve_bootstrap(precision_recall_curve, average_precision_score,
                                                                   tmp_df.Progressor.values.astype(int),
                                                                   tmp_df.Prediction.values.astype(float),
                                                                   n_bootstrap, seed, stratified=True, alpha=95)

    if color is None:
        axs[1].plot(recall, precision, label=key + f' ({np.round(ap, 2)} [{np.round(ci_l, 2)}, {np.round(ci_h, 2)}])')
    else:
        axs[1].plot(recall, precision, label=key + f' ({np.round(ap, 2)} [{np.round(ci_l, 2)}, {np.round(ci_h, 2)}])',
                    color=color)
    if legend:
        axs[1].legend()


def compute_curves_and_metrics(model_name, tmp_df, n_bootstrap=2000, seed=12345):
    auc, ci_l, ci_h, fpr, tpr = stats.calc_curve_bootstrap(roc_curve, roc_auc_score,
                                                           tmp_df.Progressor.values.astype(int),
                                                           tmp_df.Prediction.values.astype(float),
                                                           n_bootstrap, seed, stratified=True, alpha=95)

    print(f'{model_name} | AUC: {np.round(auc, 2)} [{np.round(ci_l, 2)}, {np.round(ci_h, 2)}]')

    ap, ci_l, ci_h, precision, recall = stats.calc_curve_bootstrap(precision_recall_curve, average_precision_score,
                                                                   tmp_df.Progressor.values.astype(int),
                                                                   tmp_df.Prediction.values.astype(float),
                                                                   n_bootstrap, seed, stratified=True, alpha=95)

    print(f'{model_name} | AP: {np.round(ap, 2)} [{np.round(ci_l, 2)}, {np.round(ci_h, 2)}]')

    print("=" * 80)
