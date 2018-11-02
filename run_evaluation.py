import glob
import os
import gc
import argparse
import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import cv2

import torch
from torch import nn
import torch.nn.functional as F

from sklearn.preprocessing import OneHotEncoder

from oaprogression.kvs import GlobalKVS
from oaprogression.evaluation import rstools
from oaprogression.training import model
from oaprogression.training import session as session



cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/data/DL_spring2/OA_progression_project/Data/RS_data/')
    parser.add_argument('--rs_cohort', default=3)
    parser.add_argument('--snapshots_root', default='/data/DL_spring2/OA_progression_project/snapshots')
    parser.add_argument('--snapshot', default='2018_10_21_13_44')
    args = parser.parse_args()

    mean_vect, std_vect = session.init_mean_std(args.snapshots_root, None, None, None)
    with open(os.path.join(args.snapshots_root, args.snapshot, 'session.pkl'), 'rb') as f:
        session = pickle.load(f)

    rs_meta = pd.read_csv(os.path.join(args.data_root, 'RS_metadata.csv'))
    rs_meta = rstools.preprocess_rs_meta(rs_meta, 3)

    net = model.KneeNet(session['args'][0].backbone, session['args'][0].dropout_rate)

    for fold_id in range(session['args'][0].n_folds):
        snapshot_name = glob.glob(os.path.join(args.snapshots_root, args.snapshot, f'fold_{fold_id}*.pth'))[0]

        net.load_state_dict(torch.load(snapshot_name))

        features = nn.DataParallel(net.features[:-1])
        fc = nn.DataParallel(net.classifier_prog[-1])

        features.to('cuda')
        fc.to('cuda')

        features.eval()
        fc.eval()

        preds = []
        names = []
        """
        gradcam_maps_fold = []
        for batch_id, sample in enumerate(tqdm(val_loader, total=len(loader), desc='Prediction from fold {}'.format(fold_id))):
            # We don't need gradient to make an inference  for the features
            with torch.no_grad():
                inputs = sample['I'].to("cuda")
                bs, ncrops, c, h, w = inputs.size()
                maps = features(inputs.view(-1, c, h, w))

            fc.zero_grad()
            # Registering a hook to get the gradients
            grads = []
            maps_avg = F.adaptive_avg_pool2d(maps, 1).view(maps.size(0), -1)
            # First we should attach the variable back to the graph
            maps_avg.requires_grad = True
            # Now registering the backward hook
            maps_avg.register_hook(lambda x: grads.append(x))

            # Making the inference
            # Applying the TTA right away during the forward pass
            out_tmp = F.softmax(fc(maps_avg), 1).view(bs, ncrops, -1).mean(1)
            probs_not_summed = out_tmp.to("cpu").detach().numpy()
            # Summing the probabilities values for progression
            # This allows us to predict progressor / non-progressor
            out = torch.cat((out_tmp[:, 0].view(-1, 1), out_tmp[:, 1:].sum(1).view(-1, 1)), 1)
            # Saving the results to CPU
            probs = out.to("cpu").detach().numpy()

            # Using simple one hot encoder to create a fake gradient
            ohe = OneHotEncoder(sparse=False, n_values=out.size(1))
            # Creating the fake gradient (read the paper for details)
            index = np.argmax(probs, axis=1).reshape(-1, 1)
            fake_grad = torch.from_numpy(ohe.fit_transform(index)).float().to('cuda')
            # Backward pass after which we'll have the gradients
            out.backward(fake_grad)

            # Reshaping the activation maps sand getting the weights using the stored gradients
            # This way we would be able to consider GradCAM for each crop individually

            # Making the GradCAM
            # Going over the batch
            weight = grads[-1]
            with torch.no_grad():
                weighted_A = weight.unsqueeze(-1).unsqueeze(-1).expand(*maps.size()).mul(maps)
                gcam_batch = F.relu(weighted_A).view(bs, ncrops, -1, maps.size(-2), maps.size(-1)).sum(2)
                gcam_batch = gcam_batch.to('cpu').numpy()

            gradcam_maps_fold.append(gcam_batch)
            preds.append(probs_not_summed)
            names.extend(sample['fname'])
            gc.collect()

        preds = np.vstack(preds)
        gradcam_maps_all += np.vstack(gradcam_maps_fold)
        res += preds
        gc.collect()
        """