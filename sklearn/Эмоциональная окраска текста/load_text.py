import pandas as pd
import os
import sys
import numpy as np
np.random.seed(0)

basepath = 'aclImdb'

labels = {'pos': 1, 'neg': 0}
df = pd.DataFrame()

for s in ('test', 'train'):
  for l in ('pos', 'neg'):
    path = os.path.join(basepath, s, l)
    for file in sorted(os.listdir(path)):
      with open(os.path.join(path, file), 'r', encoding='utf8') as infile:
        txt = infile.read()
      df = pd.concat([df, pd.DataFrame([[txt, labels[l]]])], ignore_index=True)
df.columns = ['review', 'sentiment']

df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv', index=False, encoding='utf-8')
