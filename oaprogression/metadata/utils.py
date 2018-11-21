from sas7bdat import SAS7BDAT
import pandas as pd
import numpy as np

def read_sas7bdata_pd(fname):
    data = []
    with SAS7BDAT(fname) as f:
        for row in f:
            data.append(row)

    return pd.DataFrame(data[1:], columns=data[0])

def data_stats(dataset):
    print('## Unique subjects', np.unique(dataset.ID).shape[0])
    print('## Knees', dataset.ID.shape[0])
    print('## Knees (left)', dataset[dataset.Side == 'L'].ID.shape[0])
    print('## Knees (right)', dataset[dataset.Side == 'R'].ID.shape[0])
    print('## Knees non-progressors', (dataset.Progressor.values == 0).sum())
    print('## Knees progressors', (dataset.Progressor.values > 0).sum())
    for KL in [0, 1, 2, 3, 4]:
        print(" ")
        print(f'### [{KL}] # knees non-progressors', (dataset[dataset.KL == KL].Progressor.values == 0).sum())
        print(f'### [{KL}] # knees progressors', (dataset[dataset.KL == KL].Progressor.values > 0).sum())
