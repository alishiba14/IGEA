import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from keras.layers import *
from keras.models import Model
from keras import backend as K
import fasttext
import numpy as np
import tensorflow as tf
import configparser
import sys
import pandas as pd
from skmultilearn.problem_transform import LabelPowerset
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
import time
from attention import Attention
from crossAttention import CrossAttention

tf.get_logger().setLevel('ERROR')



DATA_DIR = sys.argv[1]
CONFIG_PATH = sys.argv[2]

config = configparser.ConfigParser()
config.read(CONFIG_PATH)


DATASET_PATH = DATA_DIR + 'train pairs.tsv'
PREDICTSET_PATH = DATA_DIR + 'unmatched pairs.tsv'
FT_PATH = config.get('fasttext', 'location')
# Model Metadata
NUM_EPOCHS = config.getint('entity linking', 'epochs')
TRAIN_VERBOSE = config.getint('entity linking', 'train_verbose')
PREDICTION_THRESHOLD = config.getfloat('entity linking', 'prediction_threshold')
DIM_ATTENTION = config.getint('entity linking', 'attention_dimension')
DIM_LINEAR = config.getint('entity linking', 'linear_dimension')

print('entity linking with attention')
print('-loading fasttext model')
print(f'-from: {FT_PATH}')

ft_model = fasttext.load_model(FT_PATH)
embedding_dim = 300

print('-loading data')
print(f'-from {DATASET_PATH}')
data = pd.read_csv(DATASET_PATH, delimiter='\t')
tags = data['tags'].astype(str)
properties = data['properties'].astype(str)
y = data['match'].values

starttime = time.time()

# tokenize osm tags
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(list(tags.values))
text_sequences = tokenizer.texts_to_sequences(list(tags.values))

# tokenize wikidata properties
tokenizerWiki = tf.keras.preprocessing.text.Tokenizer()
tokenizerWiki.fit_on_texts(list(properties.values))
text_sequencesWiki = tokenizerWiki.texts_to_sequences(list(properties.values))

nWords =int(max(reduce(lambda count, l: count + len(l), text_sequences, 0)/len(text_sequences), reduce(lambda count, l: count + len(l), text_sequencesWiki, 0)/len(text_sequencesWiki)))


text_sequences = tf.keras.preprocessing.sequence.pad_sequences(text_sequences, maxlen=nWords, padding='post')
vocab_size = len(tokenizer.word_index) + 1
X_osm = np.array(text_sequences)

max_length = X_osm.shape[1]

weight_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    try:
        embedding_vector = ft_model[word]
        weight_matrix[i] = embedding_vector
    except KeyError:
        weight_matrix[i] = np.random.uniform(0, 0, embedding_dim)


text_sequencesWiki = tf.keras.preprocessing.sequence.pad_sequences(text_sequencesWiki, maxlen=nWords, padding='post')
vocab_sizeWiki = len(tokenizerWiki.word_index) + 1
X_wiki = np.array(text_sequencesWiki)

max_lengthWiki = X_wiki.shape[1]

weight_matrixWiki = np.zeros((vocab_sizeWiki, embedding_dim))
for word, i in tokenizerWiki.word_index.items():
    try:
        embedding_vector = ft_model[word]
        weight_matrixWiki[i] = embedding_vector
    except KeyError:
        weight_matrixWiki[i] = np.random.uniform(0, 0, embedding_dim)

def balance(x,y):
    # Import a dataset with X and multi-label y

    lp = LabelPowerset()
    ros = RandomOverSampler(random_state=42)

    # Applies the above stated multi-label (ML) to multi-class (MC) transformation.
    #yt = lp.transform(y)

    X_resampled, y_resampled = ros.fit_resample(x, y)
    # Inverts the ML-MC transformation to recreate the ML set
    #y_resampled = lp.inverse_transform(y_resampled)
    #y_resampled = y_resampled.toarray()
    return X_resampled, y_resampled


x_dist = data['dist'].values

X_osm_train, X_osm_test, y_osm_train, y_osm_test = train_test_split(X_osm, y, test_size=0.20, random_state=42)
X_wiki_train, X_wiki_test, y_wiki_train, y_wiki_test = train_test_split(X_wiki, y, test_size=0.20, random_state=42)
X_dist_train, X_dist_test, y_dist_train, y_dist_test = train_test_split(x_dist, y, test_size=0.20, random_state=42)

X_osm_train, X_osm_val, y_osm_train, y_osm_val = train_test_split(X_osm_train, y_osm_train, test_size=0.10, random_state=42)
X_wiki_train, X_wiki_val, y_wiki_train, y_wiki_val = train_test_split(X_wiki_train, y_wiki_train, test_size=0.10, random_state=42)
X_dist_train, X_dist_val, y_dist_train, y_dist_val = train_test_split(X_dist_train, y_dist_train, test_size=0.10, random_state=42)


# Note y_bal will be the same due to seeding
#x_osm_bal, y_bal = balance(X_osm_train, y_osm_train)
#x_wiki_bal, y_bal = balance(X_wiki_train, y_wiki_train)


# ------- osm tags path ------------
sentence_input = tf.keras.layers.Input(shape=(max_length,))
x = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[weight_matrix],
                              input_length=max_length)(sentence_input)
lstm = Bidirectional(LSTM(64, return_sequences = True), name="bi_lstm_0")(x)


#-----------Wikidata properties path  --------------------
sentence_inputWiki = tf.keras.layers.Input(shape=(max_lengthWiki,))
xwiki = tf.keras.layers.Embedding(vocab_sizeWiki, embedding_dim, weights=[weight_matrixWiki],
                                  input_length=max_lengthWiki)(sentence_inputWiki)


lstmwiki = Bidirectional(LSTM(64, return_sequences=True), name="bi_lstm_0wiki")(xwiki)


cross_att = CrossAttention(output_dim=64)([lstm, lstmwiki])

self_att1 = MultiHeadAttention(num_heads=2, key_dim=32)(cross_att, cross_att)
self_att1 = LayerNormalization()(self_att1)
self_att1 = Concatenate()([cross_att, self_att1])

lstm3 = Bidirectional(LSTM(32))(self_att1)

cross_att2 = CrossAttention(output_dim=64)([lstmwiki, lstm])

self_att2 = MultiHeadAttention(num_heads=2, key_dim=32)(cross_att2, cross_att2)
self_att2 = LayerNormalization()(self_att2)
self_att2 = Concatenate()([cross_att2, self_att2])

lstm4 = Bidirectional(LSTM(32))(self_att2)


sentence_input_dist = tf.keras.layers.Input(shape=(1,))
dist = Dense(1, activation="relu")(sentence_input_dist)
# ------- combine ----------
concat = tf.keras.layers.concatenate([lstm3, lstm4, dist])

concat = Dense(50, activation="relu")(concat)
#dropout = Dropout(0.05)(dense1)
output = Dense(1, activation="sigmoid")(concat)

model = tf.keras.Model(inputs=[sentence_input, sentence_inputWiki, sentence_input_dist], outputs=output)


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


METRICS = [keras.metrics.Precision(name='precision'),
           keras.metrics.Recall(name='recall')]

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc',f1_m,precision_m, recall_m])
model.summary()
model.fit([X_osm_train, X_wiki_train, X_dist_train], y_osm_train, validation_data=([X_osm_val, X_wiki_val, X_dist_val], y_osm_val), batch_size=156, epochs=NUM_EPOCHS, shuffle=True, verbose=TRAIN_VERBOSE)

#add confusion matrix

prediction = model.predict([X_osm_test, X_wiki_test, X_dist_test])
prediction = (prediction >= PREDICTION_THRESHOLD)
runtime = time.time() - starttime

with open(f'{DATA_DIR}class_report.txt', 'w', encoding='utf-8') as file:
    report = metrics.classification_report(y_osm_test, prediction)
    file.write(f'Performance Attention Model:\n')
    file.write(f'On Dataset: {DATASET_PATH}\n')
    file.write(f"runtime: {time.strftime('%H:%M:%S', time.gmtime(runtime))}\n")
    file.write(f"for {NUM_EPOCHS} epochs\n\n")
    file.write(report)
    print(report)

# save model for possible later reuse
def save_object(model, name: str):
    FILENAME = f'{DATA_DIR}{name}.sav'
    with open(FILENAME, 'wb') as file:
        pickle.dump(model, file)

model.save(DATA_DIR + 'keras model')
save_object(tokenizer, 'osm tokenizer')
save_object(tokenizerWiki, 'wikidata tokenizer')
