import pandas as pd
import sys
import configparser
from tqdm import tqdm
from json import loads
import csv

DATA_DIR = sys.argv[1]
CONFIG_PATH = sys.argv[2]

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

MATCH_FILENAME = DATA_DIR + 'train pairs.tsv'  # file to write candidate pairs to
OUTPUT_FILE_MATCH = DATA_DIR + 'keyvals.tsv'

def create_keyval_file(input: str, output: str) -> None:
    """
    Create key value pair file for embedding strategy used in embeddingKeyValue script
    Used for comparison to previous projects
    :param input:
    :param output:
    :return:
    """
    data = pd.read_csv(input, delimiter='\t')

    key_vals = []
    for index, row in tqdm(data.iterrows(), total=len(data)):
        j_dict = loads(str(row['tags']))
        osm_id = row['osm_id']
        for k, v in j_dict.items():
            if v and osm_id:
                key_vals.append({'osm_id': osm_id, 'key': k, 'value': v.replace('\t', '').replace('\n', '').replace('\r', '')})

    with open(output, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        for d in tqdm(key_vals):
            writer.writerow([d['osm_id'], d['key'], d['value']])

print('-transforming matched data for custom embeddings')
create_keyval_file(MATCH_FILENAME, OUTPUT_FILE_MATCH)
