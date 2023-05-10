import pandas as pd
import numpy as np
import fasttext
import sys
import configparser
import warnings

DATA_DIR = sys.argv[1]
CONFIG_PATH = sys.argv[2]

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

DATASET_LOCATION = DATA_DIR + 'train pairs.tsv'  # path to data file to embed (in tsv format)
PREDICTSET_LOCATION = DATA_DIR + 'unmatched pairs.tsv'
OUTPUT_PATH_TRAIN = DATA_DIR + 'el training set.parquet'  # file path for resulting file
OUTPUT_PATH_PREDICTIONS = DATA_DIR + 'el prediction set.parquet'
OUTPUT_FORMAT = 'parquet'
FT_PATH = config.get('fasttext', 'location')
SCRAPE_MODES = [s.strip() for s in config.get('wikidata scrape', 'scrape_values').split(',')]


def embed_list(labels: list, model: fasttext.FastText) -> np.ndarray:
    """
    function to create embedding from list of strings
    :param labels: list of strings (type labels)
    :param model: fasttext model for encoding
    :return: 300 dimensional array containing encoding
    """
    return model.get_word_vector(' '.join(labels)) if len(labels) > 0 else np.zeros(300)


def embed_dict(tags: dict, model: fasttext.FastText) -> np.ndarray:
    """
    legacy function to create embeddings from dictionary
    :param tags: dictionary containing key value pairs in tags
    :param model: fasttext model for encoding
    :return: 300 dimensional array containing encoding
    """
    vec = []
    if tags:
        for k, v in tags.items():
            vec.append(model.get_word_vector(k))
            vec.append(model.get_word_vector(v))
    return np.mean(vec, axis=0) if len(vec) > 0 else np.zeros(300)


def embed_tags(tags: str, model: fasttext.FastText) -> np.ndarray:
    """
    function for encoding tags from string
    expects space separated string of content to encode
    :param tags: words to encode concatenated with space
    :param model: fasttext model for embedding
    :return: 300 dimensional array containing encoding
    """
    # string will be split on spaces and mean pooled for sentences
    return model.get_word_vector(tags) if tags else np.zeros(300)

def compute_embeddings(input_path: str) -> pd.DataFrame:
    print('-reading datafile')
    print(f'-from: {input_path}')

    data = pd.read_csv(input_path, delimiter='\t')

    # strategy: just embed what there is
    columns = list(data.columns)
    drop_list = []

    if 'type' in columns:
        types = data['type'].unique()
        types_dict = {types[code]:code  for code in range(len(types))} # create dictionary of types
        data['type'] = data['type'].apply(lambda x: types_dict[x])


    # compute embeddings for wikidata type labels
    if 'labels' in columns:
        drop_list.append('labels')
        print('-embedding wikidata labels')
        data['labels'].fillna('', inplace=True)
        data = pd.concat([data, data.apply(lambda x: embed_list(str(x['labels']).split(';'), ft_model), axis=1, result_type='expand').
                         rename(columns={i: f'l_emb_{i}' for i in range(300)})], axis=1)

    # compute embeddings for osm tags
    if 'tags' in columns:
        drop_list.append('tags')
        print('-embedding osm tags')
        data['tags'].fillna('', inplace=True)
        data = pd.concat([data, data.apply(lambda x: embed_tags(x['tags'], ft_model), axis=1, result_type='expand').
                         rename(columns={i: f't_emb_{i}' for i in range(300)})], axis=1)

    # compute embeddings for osm tags
    if 'properties' in columns:
        drop_list.append('properties')
        print('-embedding wikidata properties')
        data['properties'].fillna('', inplace=True)
        data = pd.concat([data, data.apply(lambda x: embed_tags(x['properties'], ft_model), axis=1, result_type='expand').
                         rename(columns={i: f'p_emb_{i}' for i in range(300)})], axis=1)

    # usually shouldn't be here and not used for entityLinking now
    # name sim will be used
    if 'name' in columns:
        drop_list.append('name')

    # point style location not useful anymore
    if 'location' in columns:
        drop_list.append('location')

    if drop_list:
        data = data.drop(columns=drop_list, axis=1)
    return data


def write_data(data: pd.DataFrame, output_path: str) -> None:
    print('-writing transformed dataset')
    print(f'-writing to: {output_path}')
    data.to_parquet(output_path, engine='pyarrow')
    print('-writing complete')


print('embedding data')

print('-loading fasttext model')
print(f'-from: {FT_PATH}')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')  # supress warning about change of model class
    ft_model = fasttext.load_model(FT_PATH)

df = compute_embeddings(DATASET_LOCATION)
write_data(df, OUTPUT_PATH_TRAIN)

df = compute_embeddings(PREDICTSET_LOCATION)
write_data(df, OUTPUT_PATH_PREDICTIONS)
