import pandas as pd
import numpy as np
import re
import sys
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions
import csv
import configparser
from tqdm import tqdm
from urllib.error import HTTPError
import time
import socket

DATA_DIR = sys.argv[1]
CONFIG_PATH = sys.argv[2]

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

INPUT_FILE = DATA_DIR + 'osm rbf.tsv'
OUTPUT_FILE = DATA_DIR + 'nca dataset.tsv'
QID_INDEX_FILE = DATA_DIR + 'qid_index.tsv'
TESTRUN = config.getboolean('misc', 'testrun')
OSM_TAG_FILE = config.get('nca', 'osm_tag_location')
OSM_KEY_FILE = config.get('nca', 'osm_key_location')
REL_THRESHOLD = config.getint('nca',
                              'relevance_threshold')  # minimum number of class instances for class to be considered relevant

print('generating schema alignment dataset')

if TESTRUN:
    REL_THRESHOLD = 2

print('-reading osm data')

lines = []
with open(INPUT_FILE, "r", encoding='utf8') as a_file:
    for line in a_file:
        stripped_line = line.strip()
        lines.append(re.split(r'\t', stripped_line))

# rework lines with not exactly 3 tab separated entries
for i in range(len(lines)):
    try:
        if len(lines[i]) < 3:  # not a valid triplet
            lines.remove(lines[i])
    except IndexError:
        # lines.remove(lines[i]) ###remove this later
        break
    if len(lines[i]) > 4:
        del lines[i][3:]
    if len(lines[i]) == 4:
        del lines[i][3]
node = []
key = []
value = []
for i in range(len(lines)):
    try:

        node.append(lines[i][0])
        key.append(lines[i][1])
        value.append(lines[i][2])

    except IndexError:
        print(i)
for i in range(len(node)):
    node[i] = node[i].replace('<https://www.openstreetmap.org/node/', '')
    node[i] = node[i].replace('>', '')
for i in range(len(node)):
    key[i] = key[i].replace('<https://wiki.openstreetmap.org/wiki/Key:', '')
    key[i] = key[i].replace('>', '')
data = pd.DataFrame(list(zip(node, key, value)), columns=['node', 'key', 'value'])
data['value'] = data['value'].str.replace('\"', '')  # remove abundance of quotation marks
data['value'] = data['value'].str.replace('\\', '')  # remove abundance of backslashes

data['tagKey'] = data[['key', 'value']].apply(lambda x: '='.join(x), axis=1)

# get the data for tags and keys of OSM.
osmTag = pd.read_csv(OSM_TAG_FILE, sep=',', encoding='utf-8', )
osmKey = pd.read_csv(OSM_KEY_FILE, sep=',', encoding='utf-8', )

keys = list(osmKey.Keys.values)
tags = list(osmTag.Tags.values)

osm_id = []
osmwiki_id = []
osmtagkey = []
wikipedia = []
for index, row in data.iterrows():
    if row['key'] == 'wikipedia':
        wikipedia.append(row['value'])
        osmwiki_id.append(row['node'])
    if row['tagKey'] in tags:
        osm_id.append(row['node'])
        osmtagkey.append(row['tagKey'])
    else:
        osm_id.append(row['node'])
        osmtagkey.append(row['key'])

osmdata = pd.DataFrame(list(zip(osm_id, osmtagkey)), columns=['osm_id', 'osmTagKey'])
osmWiki = pd.DataFrame(list(zip(osmwiki_id, wikipedia)), columns=['osm_id', 'wikipedia'])
osmdata = pd.merge(osmWiki, osmdata, on='osm_id')

dbEnt = list(set(list(data.loc[data['key'] == 'wikipedia', 'value'])))
for i in range(len(dbEnt)):
    dbEnt[i] = dbEnt[i].replace('\"', '')


def get_results(endpoint_url, query):
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5)'
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    if TESTRUN:
        sparql.setTimeout(2)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

def transform_uri(uri: str) -> str:
    uri = uri[3:]
    uri = uri.replace(' ', '_')
    uri = uri.replace('\"', '')
    uri = uri.replace('\\', '')
    return uri

wiki_Data = []
# presort entries for faster query
print(f'-number of dbentities: {len(dbEnt)}')
entries_per_country = {}
for entry in dbEnt:
    language = entry[0:2]
    if not language in entries_per_country:
        entries_per_country.update({language: []})
    if language in ['en', 'de', 'fr']:
        entries_per_country[language].append(entry)
language_count = np.sum([len(v) for k, v in entries_per_country.items()])
print(f'-number of entries with fitting language: {language_count}')

dbpediaEnt = []
wdLabel = []
ps_Label = []
with tqdm(total=language_count, desc='-collecting dbpedia information') as pbar:
    for language, entries in entries_per_country.items():
        i = 0
        step_size = 30
        while i < len(entries):
            if language == 'en':
                endpoint_url = 'http://dbpedia.org/sparql'
                id_string = ' '.join([f"<http://dbpedia.org/resource/{transform_uri(e)}>" for e in entries[i: min(i+step_size, len(entries) - 1)]])
                query = """PREFIX db: <http://dbpedia.org/resource/>
                PREFIX prop: <http://dbpedia.org/property/>
                PREFIX onto: <http://dbpedia.org/ontology/>
                select distinct ?item ?property ?value
                where {
                    VALUES ?item {%s}
                    ?item ?property ?value.
                }""" % id_string
            else:
                endpoint_url = 'http://%s.dbpedia.org/sparql' % language
                id_string = ' '.join(f"<http://{language}.dbpedia.org/resource/{transform_uri(e)}>" for e in entries[i: min(i+step_size, len(entries)-1)])
                query = """PREFIX db: <http://%s.dbpedia.org/resource/>
                PREFIX prop: <http://%s.dbpedia.org/property/>
                PREFIX onto: <http://%s.dbpedia.org/ontology/>
                select distinct ?item ?property ?value
                where { 
                    VALUES ?item {%s}
                    ?item ?property ?value. 
                }
                """ % (language, language, language, id_string)
            try:
                results = get_results(endpoint_url, query)
                for res in results['results']['bindings']:
                    dbpediaEnt.append(f"{language}:{res['item']['value'].split('/')[-1]}")
                    wdLabel.append(res['property']['value'])
                    ps_Label.append(res['value']['value'])

            except SPARQLExceptions.QueryBadFormed as e:
                print(repr(e))
            except HTTPError as e:
                time.sleep(5)
                pbar.write(f'{i}: {repr(e)}')
                continue
            except KeyboardInterrupt as e:
                print(repr(e))
                break
            except socket.timeout:
                pbar.write(f'timeout: {i}')
                time.sleep(5)
            pbar.update(min(step_size, len(entries) - i - 1))
            i += step_size

for i in range(len(wdLabel)):
    try:
        wdLabel[i] = wdLabel[i].rsplit('/', 1)[1]
    except IndexError:
        IndexError

for i in range(len(ps_Label)):
    try:
        ps_Label[i] = ps_Label[i].rsplit('/', 1)[1]
    except IndexError:
        IndexError

wikiTable = pd.DataFrame(list(zip(dbpediaEnt, wdLabel, ps_Label)), columns=['wikipedia', 'prop', 'value'])
wikiTable = wikiTable.drop_duplicates()
wikiTable = wikiTable[wikiTable['prop'] != 'owl#sameAs']
wikiTable = wikiTable[wikiTable['prop'] != 'subject']
wikiTable = wikiTable[wikiTable['prop'] != 'wikiPageUsesTemplate']
wikiTable = wikiTable[wikiTable['prop'] != 'wikiPageWikiLink']
wikiTable = wikiTable[wikiTable['prop'] != 'rdf-schema#comment']
wikiTable = wikiTable[wikiTable['prop'] != 'abstract']
wikiTable = wikiTable[wikiTable['prop'] != 'rdf-schema#label']
wikiTable = wikiTable[wikiTable['value'] != 'France']
wikiTable = wikiTable[~wikiTable.prop.str.startswith('wikiPage')]
wikiTable = wikiTable[~wikiTable['value'].astype(str).str.match("Q[0-9]+")]

wikiTableClass = wikiTable[wikiTable['prop'] == '22-rdf-syntax-ns#type']
wikiTableClass = wikiTableClass[wikiTableClass['value'].isin(
    wikiTableClass['value'].value_counts()[wikiTableClass['value'].value_counts() > 100].index)]
wikiTableClass = wikiTableClass[~wikiTableClass['value'].isin(['human', 'owl#Thing', 'wgs84_pos#SpatialThing'])]
wikiTableClass = wikiTableClass.drop(['prop'], axis=1)
wikiTableClass = wikiTableClass.rename({'value': 'cls'}, axis=1)

wikidataTest = pd.merge(wikiTable, wikiTableClass, on='wikipedia')

wikidataTable = wikiTable[
    wikiTable['prop'].isin(list(wikidataTest['prop'].value_counts().loc[lambda x: x > 200].index))]
osmdata = pd.merge(osmdata, wikiTableClass, on='wikipedia')

cat_columns = ["osmTagKey"]
onehotTags = pd.get_dummies(osmdata, prefix_sep="_", columns=cat_columns)
onehotTags = onehotTags.groupby(['osm_id', 'wikipedia'], as_index=False).sum(numeric_only=True)

cat_columns = ["prop"]
oneHotWikiProp = pd.get_dummies(wikidataTable, prefix_sep="_", columns=cat_columns)
oneHotWikiProp = oneHotWikiProp.groupby(oneHotWikiProp['wikipedia'], as_index=False).sum(numeric_only=True)

cat_columns = ["cls"]
onehotClass = pd.get_dummies(wikiTableClass, prefix_sep="_", columns=cat_columns)
onehotClass = onehotClass.groupby(onehotClass['wikipedia'], as_index=False).sum(numeric_only=True)

tempMerge = pd.merge(oneHotWikiProp, onehotClass, on='wikipedia')
Data = pd.merge(onehotTags, tempMerge, on='wikipedia')

# save the data for the particular country and the KG
print('-saving dataset')

Data.to_csv(OUTPUT_FILE, sep='\t', encoding='utf-8', index=False)
