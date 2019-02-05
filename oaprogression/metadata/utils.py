import numpy as np
import pandas as pd
from sas7bdat import SAS7BDAT


def read_sas7bdata_pd(fname):
    data = []
    with SAS7BDAT(fname) as f:
        for row in f:
            data.append(row)

    return pd.DataFrame(data[1:], columns=data[0])


def data_stats(dataset, participants):
    dataset = pd.merge(dataset, participants, on=('ID', 'Side'))

    print('## Unique subjects', np.unique(dataset.ID).shape[0])
    print('## Males', np.unique(dataset[dataset.SEX == 1].ID).shape[0])
    print('## Females', np.unique(dataset[dataset.SEX == 0].ID).shape[0])

    print('## Mean Age', np.nanmean(participants.AGE))
    print('## STD Age', np.nanstd(participants.AGE))

    print('## Mean BMI', np.nanmean(participants.BMI))
    print('## STD BMI', np.nanstd(participants.BMI))

    print('## Knees', dataset.ID.shape[0])

    print('## Knees (left non-progressors)',
          dataset[(dataset.Side == 'L') & (dataset.Progressor.values == 0)].ID.shape[0])
    print('## Knees (right non-progressors)',
          dataset[(dataset.Side == 'R') & (dataset.Progressor.values == 0)].ID.shape[0])

    print('## Knees (left progressors)',
          dataset[(dataset.Side == 'L') & (dataset.Progressor.values > 0)].ID.shape[0])
    print('## Knees (right progressors)',
          dataset[(dataset.Side == 'R') & (dataset.Progressor.values > 0)].ID.shape[0])

    print('## Knees non-progressors', (dataset.Progressor.values == 0).sum())
    print('## Knees progressors', (dataset.Progressor.values > 0).sum())

    print('## Knees non-progressors (males)', (dataset[dataset.SEX == 1].Progressor.values == 0).sum())
    print('## Knees progressors (males)', (dataset[dataset.SEX == 1].Progressor.values > 0).sum())

    print('## Knees non-progressors (females)', (dataset[dataset.SEX == 0].Progressor.values == 0).sum())
    print('## Knees progressors (females)', (dataset[dataset.SEX == 0].Progressor.values > 0).sum())

    for KL in [0, 1, 2, 3, 4]:
        print(" ")
        print(f'### [{KL}] # knees non-progressors', (dataset[dataset.KL == KL].Progressor.values == 0).sum())
        print(f'### [{KL}] # knees progressors', (dataset[dataset.KL == KL].Progressor.values > 0).sum())
