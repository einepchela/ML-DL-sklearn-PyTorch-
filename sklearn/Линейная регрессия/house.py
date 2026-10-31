import pandas as pd

columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area',
           'Central Air', 'Total Bsmt SF', 'SalePrice']
df = pd.read_csv(r'C:\Users\User\AppData\Local\Programs\Python\Python314\AmesHousing.txt', sep='\t',
                 usecols=columns)
df.head()

df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})
df = df.dropna(axis=0)
