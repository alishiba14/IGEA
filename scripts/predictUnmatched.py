import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
import pandas as pd
import sys
import configparser
import asyncio
import asyncpg
import tensorflow as tf
import keras
import numpy as np
from attention import Attention
from crossAttention import CrossAttention

tf.get_logger().setLevel('ERROR')
DATA_DIR = sys.argv[1]
CONFIG_PATH = sys.argv[2]
ITERATION = int(sys.argv[3])

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

USE_ATTENTION = config.getboolean('entity linking', 'attention')
if not USE_ATTENTION:
    DATASET_LOCATION = DATA_DIR + 'el prediction set.parquet'  # location of parquet file for unmatched candidate pairs
    MODEL_TYPE = config.get('entity linking', 'model')
    CLASSIFIER_LOCATION = DATA_DIR + MODEL_TYPE + '.sav'  # location of classifier model saved to pickle file
else:
    DATASET_LOCATION = DATA_DIR + 'unmatched pairs.tsv'
    CLASSIFIER_LOCATION = DATA_DIR + 'self attention.sav'
    OSM_TOKENIZER_LOCATION = DATA_DIR + 'osm tokenizer.sav'
    WIKIDATA_TOKENIZER_LOCATION = DATA_DIR + 'wikidata tokenizer.sav'

OUTPUT_PATH = DATA_DIR + 'predicted entity matches.tsv'
PREDICTION_THRESHOLD = config.getfloat('entity linking', 'prediction_threshold')

# postGIS config
PW_FILENAME = config.get('postGIS', 'passwordfile')
PG_HOST = config.get('postGIS', 'host')
PG_USER = config.get('postGIS', 'user')
PG_DB_NAME = config.get('postGIS', 'dbname')
PG_PORT = config.getint('postGIS', 'port')
TABLE_NAME = config.get('entity linking', 'prediction_table')

print('predicting entity matches')
print('-loading dataset of possible new matchings')
print(f'-from: {DATASET_LOCATION}')

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

if USE_ATTENTION:
    data = pd.read_csv(DATASET_LOCATION, delimiter='\t')

    print('-loading classifier')
    print(f'-from: {CLASSIFIER_LOCATION}')
    model = keras.models.load_model(DATA_DIR + 'keras model', custom_objects={"CrossAttention": CrossAttention, "f1_m": f1_m, "precision_m": precision_m, "recall_m": recall_m})
    nWords = model.layers[0].get_output_at(0).get_shape()[1]
    print('-loading osm tokenizer')
    print(f'-from: {OSM_TOKENIZER_LOCATION}')
    with open(OSM_TOKENIZER_LOCATION, 'rb') as file:
        osm_tokenizer = pickle.load(file)

    print('-loading wikidata tokenizer')
    print(f'-from: {WIKIDATA_TOKENIZER_LOCATION}')
    with open(WIKIDATA_TOKENIZER_LOCATION, 'rb') as file:
        wiki_tokenizer = pickle.load(file)

    tags = data['tags'].astype(str)
    properties = data['properties'].astype(str)
    x_dist = data['dist'].values
    text_sequences_osm = osm_tokenizer.texts_to_sequences(list(tags.values))
    text_sequences_osm = tf.keras.preprocessing.sequence.pad_sequences(text_sequences_osm, maxlen=nWords, padding='post')
    x_osm = np.array(text_sequences_osm)

    text_sequences_wiki = wiki_tokenizer.texts_to_sequences(list(properties.values))
    text_sequences_wiki = tf.keras.preprocessing.sequence.pad_sequences(text_sequences_wiki, maxlen=nWords, padding='post')
    x_wiki = np.array(text_sequences_wiki)

    print(f'-predicting matches with threshold: {PREDICTION_THRESHOLD}')
    probabilities = model.predict([x_osm, x_wiki, x_dist])
    prediction = (probabilities >= PREDICTION_THRESHOLD)
else:
    data = pd.read_parquet(DATASET_LOCATION, engine='pyarrow')

    print('-loading classifier')
    print(f'-from: {CLASSIFIER_LOCATION}')
    with open(CLASSIFIER_LOCATION, 'rb') as file:
        model = pickle.load(file)

    print(f'-predicting matches with threshold: {PREDICTION_THRESHOLD}')
    predicted_values = model.predict_proba(data.iloc[:, 3:])
    probabilities = predicted_values[:, 1]
    prediction = (probabilities >= PREDICTION_THRESHOLD)

print(f'-Number of predicted matches: {int(prediction.sum())} / {len(prediction)}')
print(f'-matched percentage: {(int(prediction.sum()) / len(prediction)) * 100: .2f}%')

prediction_pairs = pd.DataFrame()
prediction_pairs['wkid'] = data['wkid']
prediction_pairs['osm_id'] = data['osm_id']
prediction_pairs['probability'] = probabilities
prediction_pairs['prediction'] = prediction

print('-logging predictions for matches')
print(f'-to: {OUTPUT_PATH}')
with open(OUTPUT_PATH, 'w', encoding='utf-8', newline='') as file:
    prediction_pairs.to_csv(file, sep='\t')
print('-writing complete')

prediction_pairs = prediction_pairs[prediction]

# update predictions in database
with open(PW_FILENAME, 'r') as file:
    password = file.read().strip()

async def generate_list(df: pd.DataFrame) -> tuple:
    entries = []
    for index, row in df.iterrows():
        entries.extend([row[0], row[1], str(row[2]), ITERATION])
    return tuple(entries)

async def update_predictions():
    conn = await asyncpg.connect(
        user=PG_USER,
        password=password,
        database=PG_DB_NAME,
        host=PG_HOST,
        port=PG_PORT
    )

    sql = f"INSERT INTO {TABLE_NAME} (wkid, osm_id, confidence, iteration) VALUES "
    i = 0
    batchsize = 2

    async with conn.transaction():
        while i < len(prediction_pairs) - 1:
            offset = min(batchsize, len(prediction_pairs) - 1 - i)
            inserts = ','.join(["($1, $2, $3, $4)"] * offset)
            await conn.execute(sql + inserts, *generate_list(prediction_pairs[i:i + offset]))
            i += offset

    await conn.close()

print(f'-writing matches to {TABLE_NAME}')
asyncio.run(update_predictions())
print(f'-writing complete')
