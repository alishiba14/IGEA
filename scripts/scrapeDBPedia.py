import re
from SPARQLWrapper import SPARQLWrapper
from SPARQLWrapper import SPARQLExceptions
from urllib.error import HTTPError
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import sys
import configparser
import time
import socket

DATA_DIR = sys.argv[1]
CONFIG_PATH = sys.argv[2]

config = configparser.ConfigParser()
config.read(CONFIG_PATH)


OUTPUT_PATH = DATA_DIR + 'wikidata dump.parquet'
CLASSFILE_PATH = DATA_DIR + 'wikidata classes.txt'
TESTRUN = config.getboolean('misc', 'testrun')
LIMIT = config.getint('misc', 'limit')
DBPEDIA_SOURCE = config.get('dbpedia scrape', 'dbpedia_source')
DBPEDIA_COUNTRY = config.get('dbpedia scrape', 'country')

if DBPEDIA_SOURCE == 'en':
    sparql = SPARQLWrapper(f"http://dbpedia.org/sparql",
                       returnFormat='json',
                       agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5)')
else:
    sparql = SPARQLWrapper(f"http://{DBPEDIA_SOURCE}.dbpedia.org/sparql",
                       returnFormat='json',
                       agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5)')

print('-reading classes')
print(f'-from: {CLASSFILE_PATH}')
linked_classes = []
with open(CLASSFILE_PATH, 'r', encoding='utf-8') as file:
    for line in file.readlines():
        text = line.strip().replace('\n', '')
        if text:
            linked_classes.append(text)

# collect geo entities for classes and initial data
query = """
PREFIX de: <http://de.dbpedia.org/resource/>
PREFIX fr: <http://fr.dbpedia.org/resource/>
PREFIX country: <http://dbpedia.org/resource/>
PREFIX db: <http://dbpedia.org/resource/>
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX dbo: <http://dbpedia.org/ontology/>

SELECT ?item AVG(?lat) as ?lat AVG(?lon) as ?lon WHERE {
    ?item rdf:type dbo:%s.
    ?item dbo:country country:%s.
    ?item geo:lat ?lat.
    ?item geo:long ?lon
} group by ?item
"""

entities = {}

with tqdm(total=len(linked_classes), desc=f'-Gathering entities in ({DBPEDIA_COUNTRY})', miniters=1) as pbar:
    for clazz in linked_classes:
        try:
            pbar.set_postfix_str(f'class: {clazz}')
            sparql.setQuery(query % (clazz, DBPEDIA_COUNTRY))
            results = sparql.query().convert()
            for res in results['results']['bindings']:
                id = res['item']['value'].split('/')[-1]
                entities.update({id: {'wkid': id, 'location': f"Point({res['lon']['value']} {res['lat']['value']})", 'type': clazz}})
        except SPARQLExceptions.QueryBadFormed as e:
            print(repr(e))
        except HTTPError as e:
            time.sleep(5)
            pbar.write(f'{clazz}: {repr(e)}')
            continue
        except KeyboardInterrupt as e:
            print(repr(e))
            break
        except socket.timeout:
            pbar.write(f'timeout: {clazz}')
            time.sleep(5)
        pbar.update(1)
        if TESTRUN:
            if len(entities) > LIMIT:
                pbar.write(f'Test limit {LIMIT} reached')
                break

if TESTRUN:
    entities = dict(list(entities.items())[:LIMIT])

def filter_key_value_pairs(pairs: list) -> str:
    """
    filter list of key value tuples and transform into single string
    :param pairs: list of key value tuples
    :return: concatenation of filtered key value pairs
    """
    filtered_items = []
    key_filter = ['owl#sameAs', 'subject', 'wikiPageUsesTemplate', 'wikiPageWikiLink', 'rdf-schema#comment', 'abstract', 'rdf-schema#label']
    value_filter = ['France']
    for k, v in pairs:
        if k in key_filter:
            # remove meta information
            pass
        elif str(k).startswith('wikiPage'):
            pass
        elif v in value_filter:
            pass
        elif re.match(r"Q[0-9]+", str(v)):
            # corresponding wikidata id
            pass
        else:
            # handle underscores from uri format
            key = str(k).replace('_', ' ')
            value = str(v).replace('_', ' ')
            filtered_items.extend([key, value])
    return ' '.join(filtered_items)

property_query = """
PREFIX de: <http://de.dbpedia.org/resource/>
PREFIX fr: <http://fr.dbpedia.org/resource/>
PREFIX db: <http://dbpedia.org/resource/>
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX dbo: <http://dbpedia.org/ontology/>

select distinct ?item ?property ?value
    where {
        VALUES ?item {%s}
        ?item ?property ?value.
    }
"""

for k, v in entities.items():
    v.update({'properties': ''})

i = 0
step_size = 70
with tqdm(total=len(entities), desc='-updating properties') as pbar:
    while i < len(entities):
        id_string = ' '.join(f"<http://dbpedia.org/resource/{e}>" for e in list(entities.keys())[i:min(i+step_size, len(entities)-1)])
        try:
            sparql.setQuery(property_query % (id_string))
            results = sparql.query().convert()
            properties = {}
            for res in results['results']['bindings']:
                id = res['item']['value'].split('/')[-1]
                if id not in properties:
                    properties.update({id: []})

                key = res['property']['value'].split('/')[-1]
                value = res['value']['value'].split('/')[-1]
                properties[id].append((key, value))
            for k, v in properties.items():
                entities[k].update({'properties': filter_key_value_pairs(v)})
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
        i += step_size
        pbar.update(step_size)

def write_to_file(entity_dict: dict, filename: str) -> None:
    """
    function to transform entity dictionary into dataframe for saving in parquet format
    :param entity_dict: dictionary containing wikidata information
    :param filename: path to write file to
    :return:
    """
    data_list = list(entity_dict.values())
    table = pa.Table.from_pylist(data_list)
    with open(filename, 'wb') as file:
        pq.write_table(table, file)

print('-writing scraped data')
print(f'-to: {OUTPUT_PATH}')
write_to_file(entities, OUTPUT_PATH)
print('-writing complete')
