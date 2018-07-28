'''
Label the blobs using a previously trained model

Gary Bishop July 2018
'''

import pandas as pd
import Args
import pickle
from features import getFeatures

args = Args.Parse(
    inblobs='output.blobs.bz2',
    outblobs='output.labeled.bz2',
    model='models/LR1.pkl'
)

data = pd.read_pickle(args.inblobs)

model = pickle.load(open(args.model, 'rb'))


features = getFeatures(data)

labels = model.predict(features)
data.isdot = labels

data.to_pickle(args.outblobs)
