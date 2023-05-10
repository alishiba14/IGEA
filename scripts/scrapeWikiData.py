from SPARQLWrapper import SPARQLWrapper
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import sys
import configparser

DATA_DIR = sys.argv[1]
CONFIG_PATH = sys.argv[2]

config = configparser.ConfigParser()
config.read(CONFIG_PATH)


OUTPUT_PATH = DATA_DIR + 'wikidata dump.parquet'
CLASSFILE_PATH = DATA_DIR + 'wikidata classes.txt'
TESTRUN = config.getboolean('misc', 'testrun')
LIMIT = config.getint('misc', 'limit')
COUNTRY_ID = config.get('wikidata scrape', 'country_id')
SCRAPE_MODES = [s.strip() for s in config.get('wikidata scrape', 'scrape_values').split(',')]
NAME_LANGUAGE = config.get('wikidata scrape', 'name_language')

sparql = SPARQLWrapper("https://query.wikidata.org/sparql",
                       returnFormat='json',
                       agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5)')

print('reading classes')
print(f'from: {CLASSFILE_PATH}')
linked_classes = []
with open(CLASSFILE_PATH, 'r', encoding='utf-8') as file:
    for line in file.readlines():
        text = line.strip().replace('\n', '')
        if text:
            linked_classes.append(text)

# collect geo entities for classes and initial data
query = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?item ?type ?location WHERE {
    ?item wdt:P17 wd:%s. # in Germany
    ?item wdt:P31* wd:%s. # instance or subinstance of (to be specified)
    ?item wdt:P625 ?location.
    FILTER (strstarts(str(?location), 'Point'))
}
"""

entities = {}

with tqdm(total=len(linked_classes), desc=f'-Gathering entities in ({COUNTRY_ID})', miniters=1) as pbar:
    for clazz in linked_classes:
        try:
            pbar.set_postfix_str(f'class: {clazz}')
            sparql.setQuery(query % (COUNTRY_ID, clazz))
            results = sparql.query().convert()
            for res in results['results']['bindings']:
                wkid = res['item']['value'].split('/')[-1]
                entities.update({wkid: {'wkid': wkid, 'location': res['location']['value'], 'type': clazz}})
        except:
            pbar.write(f'An error occurred gathering {clazz}, skipping...')
        pbar.update(1)
        if TESTRUN:
            if len(entities) > LIMIT:
                break

if TESTRUN:
    entities = dict(list(entities.items())[:LIMIT])

# update popularity
if 'popularity' in SCRAPE_MODES:
    pop_query = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    
    SELECT ?kgentity  (count(distinct ?p) as ?prop) ?location {
      VALUES ?kgentity {%s}
      ?kgentity ?p ?statement .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    } group by ?kgentity ?location
    """

    # set defaults
    for k, v in entities.items():
        v.update({'pop': 0})

    i = 0
    step_size = 300
    with tqdm(total=len(entities), desc='-updating popularity') as pbar:
        while i < len(entities):
            id_string = ''.join(f"wd:{e} " for e in list(entities.keys())[i:min(i+step_size, len(entities)-1)])
            sparql.setQuery(pop_query % id_string)
            results = sparql.query().convert()
            for res in results['results']['bindings']:
                entities[res['kgentity']['value'].split('/')[-1]].update({'pop': int(res['prop']['value'])})
            i += step_size
            pbar.update(step_size)



# add type labels for entities
if 'type labels' in SCRAPE_MODES:
    label_query = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    
    SELECT ?kgentity (GROUP_CONCAT(distinct ?typeLabel; SEPARATOR = "; ") as ?labels) ?location {
      VALUES ?kgentity {%s}
      ?kgentity wdt:P31 ?label.
      ?label rdfs:label ?typeLabel.
      FILTER(lang(?typeLabel) = 'en').
    } group by ?kgentity ?location
    """

    # set defaults
    for k, v in entities.items():
        v.update({'labels': ''})

    i = 0
    step_size = 300

    with tqdm(total=len(entities), desc='-updating labels', miniters=1) as pbar:
        while i < len(entities):
            id_string = ''.join(f"wd:{e} " for e in list(entities.keys())[i:min(i+step_size, len(entities)-1)])
            sparql.setQuery(label_query % id_string)
            results = sparql.query().convert()
            for res in results['results']['bindings']:
                entities[res['kgentity']['value'].split('/')[-1]].update({'labels': res['labels']['value']})
            i += step_size
            pbar.update(step_size)

# add names to entities
if 'name' in SCRAPE_MODES:
    name_query = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    
    SELECT ?kgentity ?kgentityLabel {
        VALUES ?kgentity {%s}
        SERVICE wikibase:label { bd:serviceParam wikibase:language '%s', 'en'}
    }
    """

    # set defaults
    for k, v in entities.items():
        v.update({'name': ''})

    i = 0
    step_size = 300

    with tqdm(total=len(entities), desc='-updating names', miniters=1) as pbar:
        while i < len(entities):
            id_string = ''.join(f"wd:{e} " for e in list(entities.keys())[i:min(i+step_size, len(entities)-1)])
            sparql.setQuery(name_query % (id_string, NAME_LANGUAGE))
            results = sparql.query().convert()
            for res in results['results']['bindings']:
                entities[res['kgentity']['value'].split('/')[-1]].update({'labels': res['kgentityLabel']['value']})
            i += step_size
            pbar.update(step_size)


# add full properties per entity
if 'full properties' in SCRAPE_MODES:
    property_query = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    
    SELECT ?kgentity ?kgentityLabel ?wdLabel ?ps_Label {
        VALUES ?kgentity {%s}
        ?kgentity ?p ?statement .
        ?statement ?ps ?ps_ .
  
        ?wd wikibase:claim ?p.
        ?wd wikibase:statementProperty ?ps.
  
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    } ORDER BY ?kgentity
    """

    # set defaults
    for k, v in entities.items():
        v.update({'properties': ''})

    i = 0
    step_size = 250

    with tqdm(total=len(entities), desc='-updating properties', miniters=1) as pbar:
        while i < len(entities):
            try:
                id_string = ''.join(f"wd:{e} " for e in list(entities.keys())[i:min(i+step_size, len(entities)-1)])
                sparql.setQuery(property_query % id_string)
                results = sparql.query().convert()
                cur_id = ''
                property_pairs = set()
                for res in results['results']['bindings']:
                    wkid = res['kgentity']['value'].split('/')[-1]
                    if cur_id: # skip first test
                        if cur_id != wkid:
                            entities[cur_id].update({'properties': ' '.join(property_pairs)})
                            property_pairs = set()
                            property_pairs.add(f"{'label'} {res['kgentityLabel']['value']}")
                    else:
                        property_pairs.add(f"{'label'} {res['kgentityLabel']['value']}")
                    property_pairs.add(f"{res['wdLabel']['value']} {res['ps_Label']['value']}")
                    cur_id = wkid
                entities[cur_id].update({'properties': ' '.join(property_pairs)})
            except:
                pbar.write(f'An error occurred gathering {i} - {i + step_size}, skipping...')
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
