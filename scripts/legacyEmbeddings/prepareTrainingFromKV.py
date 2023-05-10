import pandas as pd
import numpy as np
import sys

DATA_DIR = sys.argv[1]

OUTPUT_PATH = DATA_DIR + 'el training set.parquet'
DATA_PATH = DATA_DIR + 'train pairs.tsv'
EMBEDDINGS_PATH = DATA_DIR + 'custom embeddings.csv'

data = pd.read_csv(DATA_PATH, delimiter='\t')
embeddings = pd.read_csv(EMBEDDINGS_PATH, delimiter=' ')
merged = data.join(embeddings, lsuffix=1, rsuffix=0)

print('merging data with custom embeddings')

merged.columns = list(data.columns) + ['osmid2'] + [f'emb_{i}' for i in range(300)]

types = merged['type'].unique()
types_dict = {types[code]: code for code in range(len(types))}  # create dictionary of types
merged['type'] = merged['type'].apply(lambda x: types_dict[x])

print('-dropping unused columns')

column_mask = [c not in ['osm_id', 'wkid', 'pop', 'type', 'match'] for c in data.columns]
dropcolumns = list(np.array(list(data.columns))[column_mask]) + ['osmid2']
merged = merged.drop(columns=dropcolumns, axis=1,)

# fill unmatched embeddings
merged = merged.fillna(0)

merged.to_parquet(OUTPUT_PATH, engine='pyarrow')
