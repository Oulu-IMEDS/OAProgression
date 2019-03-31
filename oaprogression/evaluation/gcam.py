import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import solt.core as slc
import solt.transforms as slt
import torch
import torch.nn.functional as F
import torchvision.transforms as tv_transforms
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from oaprogression.training.dataset import unpack_solt_data, img_labels2solt


def eval_batch(sample, features, fc, fc_kl=None):
    # We don't need gradient to make an inference  for the features
    with torch.no_grad():
        inputs = sample['img'].to("cuda")
        bs, ncrops, c, h, w = inputs.size()
        maps = features(inputs.view(-1, c, h, w))
        maps_avg = F.adaptive_avg_pool2d(maps, 1).view(maps.size(0), -1)
        if fc_kl is not None:
            out_kl = F.softmax(fc_kl(maps_avg), 1).view(bs, ncrops, -1).mean(1)
            out_kl = out_kl.to('cpu').numpy()
    fc.zero_grad()
    # Registering a hook to get the gradients
    grads = []

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
    ohe = OneHotEncoder(sparse=False, categories=[range(out.size(1))])
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

    if fc_kl is not None:
        return gcam_batch, probs_not_summed, out_kl
    return gcam_batch, probs_not_summed


def preds_and_hmaps(results, gradcams, dataset_root, figsize, threshold, savepath, gcam_type='prog'):
    if gcam_type not in ['prog', 'non-prog']:
        raise ValueError('gcam type shoould be either prog or non-prog')

    ids_rs = []
    hmaps = []

    w, h = 700, 700  # 140x140mm
    size = (650, 650)  # 130x130mm - were used in the evaluation
    x1 = w // 2 - size[0] // 2
    y1 = h // 2 - size[1] // 2

    gcam_trf = tv_transforms.Compose([
        img_labels2solt,
        slc.Stream([
            slt.PadTransform(pad_to=(700, 700), padding='z'),
            slt.CropTransform(crop_size=(700, 700), crop_mode='c'),
        ], interpolation='bicubic'),
        unpack_solt_data,
    ])

    for i, entry in tqdm(results.iterrows(), total=results.shape[0], desc=f'GradCAM [{gcam_type}]:'):
        if gcam_type == 'prog':
            if entry.pred < threshold or entry.Progressor == 0:
                continue
        else:
            if entry.pred > threshold or entry.Progressor == 1:
                continue

        img = cv2.imread(os.path.join(dataset_root, f'{entry.ID}_00_{entry.Side}.png'), 0)

        if 'L' == entry.Side:
            img = cv2.flip(img, 1)

        img = img.reshape((img.shape[0], img.shape[1], 1))
        img, _, _ = gcam_trf((img, 0, 0))
        img = img.squeeze()

        # We had 310x310 image and 5 300x300 crops
        # Now we map these crops back to the image
        tmp = np.zeros((h, w))
        # Center crop
        tmp[y1:y1 + size[0], x1:x1 + size[1]] += cv2.resize(gradcams[i, 0, :, :], size)
        # Upper-left crop
        tmp[0:size[0], 0:size[1]] += cv2.resize(gradcams[i, 1, :, :], size)
        # Upper-right crop
        tmp[0:size[0], w - size[1]:w] += cv2.resize(gradcams[i, 2, :, :], size)
        # Bottom-left crop
        tmp[h - size[0]:h, 0:size[1]] += cv2.resize(gradcams[i, 3, :, :], size)
        # Bottom-right crop
        tmp[h - size[0]:h, w - size[1]:w] += cv2.resize(gradcams[i, 4, :, :], size)

        tmp -= tmp.min()
        tmp /= tmp.max()
        tmp *= 255

        hmaps.append(tmp)
        ids_rs.append(entry.ID)

        plt.figure(figsize=(figsize, figsize))
        plt.subplot(121)
        plt.title(f'Original Image {entry.ID}')
        plt.imshow(img, cmap=plt.cm.Greys_r)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(122)
        plt.title(f'Prog. {entry.KL} -> {entry.KL + entry.Prog_increase} | {entry.Progressor_type}')
        plt.imshow(img, cmap=plt.cm.Greys_r)
        plt.imshow(tmp, cmap=plt.cm.jet, alpha=0.5)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(savepath, f'{entry.ID}_{entry.Side}.pdf'), bbox_inches='tight')
        plt.close()
