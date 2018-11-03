import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import torch


def eval_batch(sample, features, fc, ):
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

    return gcam_batch, probs_not_summed