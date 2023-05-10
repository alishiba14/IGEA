import pandas as pd
import numpy as np
import re
import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import csv
import configparser
from tqdm import tqdm

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
REL_THRESHOLD = config.getint('nca', 'relevance_threshold')  # minimum number of class instances for class to be considered relevant

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
        if len(lines[i]) < 3: # not a valid triplet
            lines.remove(lines[i])
    except IndexError:
        #lines.remove(lines[i]) ###remove this later
        break
    if len(lines[i]) > 4:
        del lines[i][3:]
    if len(lines[i]) == 4:
        del lines[i][3]

# write line content to lists containing triplets
node = []
key = []
value = []
for i in range(len(lines)):
    node.append(lines[i][0])
    key.append(lines[i][1])
    value.append(lines[i][2])

# remove osm_id prefix
for i in range(len(node)):
    node[i] = node[i].replace('<https://www.openstreetmap.org/node/', '')
    node[i] = node[i].replace('>', '')

# remove key prefix
for i in range(len(node)):
    key[i] = key[i].replace('<https://wiki.openstreetmap.org/wiki/Key:', '')
    key[i] = key[i].replace('>', '')


data = pd.DataFrame(list(zip(node, key, value)), columns=['node', 'key', 'value'])

data['value'] = data['value'].str.replace('\"', '')  # remove abundance of quotation marks

data['tagKey'] = data[['key', 'value']].apply(lambda x: '='.join(x), axis=1)


#data = data[(data.key != '<http://www.w3.org/2003/01/geo/wgs84_pos#long') & (data.key != '<http://www.w3.org/2003/01/geo/wgs84_pos#Point')]
#data = data[(data.key != '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type') & (data.key != '<http://www.w3.org/2003/01/geo/wgs84_pos#lat')]


#get the data for tags and keys of OSM.
osmTag = pd.read_csv(OSM_TAG_FILE, sep=',', encoding='utf-8',)
osmKey = pd.read_csv(OSM_KEY_FILE, sep=',', encoding='utf-8',)


osmKey = osmKey.drop_duplicates(subset='Keys', keep="first")


keys = list(osmKey.Keys.values)
tags = list(osmTag.Tags.values)


#create tags for key-value pair
osm_id = []
osmwiki_id = []
osmtagkey = []
osmvalue = []
wikidata = []
for index, row in data.iterrows():
    if row['key'] == 'wikidata':
        wikidata.append(row['value'])
        osmwiki_id.append(row['node'])
    if row['tagKey'] in tags:
        osm_id.append(row['node'])
        osmtagkey.append(row['tagKey'])
        osmvalue.append(row['value'])
    else:
        osm_id.append(row['node'])
        osmtagkey.append(row['key'])
        osmvalue.append(row['value'])        


osmdata = pd.DataFrame(list(zip(osm_id, osmtagkey, osmvalue)), columns=['osm_id', 'osmTagKey', 'value'])


osmWiki = pd.DataFrame(list(zip(osmwiki_id, wikidata)), columns=['osm_id', 'wikidata'])


osmdata = pd.merge(osmWiki, osmdata, on='osm_id')


wikiEnt = list(set(list(data.loc[data['key'] == 'wikidata', 'value'])))
for i in range(len(wikiEnt)):
    wikiEnt[i] = wikiEnt[i].replace('\"', '')
#remove values which do not have wikidata format: Q----
regex = re.compile('(Q)[0-9]+')
wikiEnt = [x for x in wikiEnt if regex.match(x)]


# read wikidata information for extracted qids
i = 0
wiki_Data = []
user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
# TODO adjust user agent; see https://w.wiki/CX6
sparql = SPARQLWrapper("https://query.wikidata.org/sparql",
                       agent=user_agent,
                       returnFormat='json')
with tqdm(total=len(wikiEnt), desc='-collecting wikidata information') as pbar:
    query = """SELECT ?kgentity  ?wdLabel ?ps_ ?ps_Label {
      VALUES ?kgentity {wd:%s}
      ?kgentity ?p ?statement .
      ?statement ?ps ?ps_ .
      
      ?wd wikibase:claim ?p.
      ?wd wikibase:statementProperty ?ps.
      
      OPTIONAL {
      ?statement ?pq ?pq_ .
      ?wdpq wikibase:qualifier ?pq .
      }
      
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    } ORDER BY ?wd ?statement ?ps_"""

    while i < len(wikiEnt):
        iterstep = min(len(wikiEnt) - 1, i+300)
        mystring = ''.join('wd:{0} '.format(w) for w in wikiEnt[i: iterstep])
        try:
            sparql.setQuery(query % mystring)
            results = sparql.query().convert()
            wiki_Data.append(results)
        except:  # use general except due to many possible problems with wikidata
            pbar.write(f'error collecting {i} to {iterstep}')
        pbar.update(300)
        i += 300


kgentity = []
wdLabel = []
ps_Label = []
for result in tqdm(wiki_Data, desc='-processing entities'):
    for bindings in result['results']['bindings']:
        kgentity.append(bindings['kgentity']['value'])
        wdLabel.append(bindings['wdLabel']['value'])
        ps_Label.append(bindings['ps_Label']['value'])


for i in range(len(kgentity)):
    kgentity[i] = kgentity[i].replace('http://www.wikidata.org/entity/', '')


wikidataTable = pd.DataFrame(list(zip(kgentity, wdLabel, ps_Label)), columns=['wikidata', 'prop', 'value'])


wikidataTable = wikidataTable.drop_duplicates()


wikiForClass = []  # wikidata ids with class
temp = []
cls = []  # class names
cls_qids = []
wikidata = []  # wikidata ids
for result in tqdm(wiki_Data, desc='-processing classes'):
    for bindings in result['results']['bindings']:
        temp.append(bindings['wdLabel']['value'])
        wikidata.append(bindings['kgentity']['value'].replace('http://www.wikidata.org/entity/', ''))
        if (bindings['wdLabel']['value'] == 'instance of'):
            cls.append(bindings['ps_Label']['value'])
            cls_qids.append(bindings['ps_']['value'].replace('http://www.wikidata.org/entity/', ''))
            wikiForClass.append(bindings['kgentity']['value'].replace('http://www.wikidata.org/entity/', ''))


# keep unique qid, name combinations for disambiguation later
qid_index = [[cls_name, cls_qid] for cls_name, cls_qid in set(zip(cls, cls_qids))]

wikidatacls = pd.DataFrame(list(zip(wikiForClass, cls)), columns=['wikidata', 'cls'])


# only use classes with more than REL_THRESHOLD entities
wikidataToConsider = wikidatacls[wikidatacls['cls'].isin(wikidatacls['cls'].value_counts()[wikidatacls['cls'].value_counts() > REL_THRESHOLD].index)]


wikidataTest = pd.merge(wikidataTable,wikidataToConsider, on='wikidata')


tfidf = []
className = []
propName = []
for j in tqdm(list(wikidataTest['cls'].unique()), desc='-processing tf idf'):
    if j == 'human':
        continue
    else:
        for i in (list(wikidataTest[wikidataTest['cls'] == j]['prop'].unique())):
            tf = len(wikidataTest[(wikidataTest['cls'] == j) & (wikidataTest['prop'] == i)])
            df = len(wikidataTest[wikidataTest['prop'] == i]['cls'].value_counts())
            N = 40
            weight = tf * (np.log (N/df))
            if weight == 0:
                continue
            else:
                tfidf.append(weight)
                className.append(j)
                propName.append(i)


tfidfweights = pd.DataFrame(list(zip(className, propName, tfidf)),
                    columns = ['cls', 'prop', 'tfidfval'])


groupsort = tfidfweights.sort_values(['cls'], ascending=True).groupby(['cls'], sort=False).apply(lambda x: x.sort_values(['tfidfval'], ascending=False)).reset_index(drop=True)

groupsort = groupsort.groupby('cls').head(25)
currentList = list(groupsort.prop.unique())
wikidataTable = wikidataTable[wikidataTable['prop'].isin(currentList)]
osmdata = pd.merge(osmdata, wikidataToConsider, on='wikidata')

cat_columns = ["osmTagKey"]
onehotTags = pd.get_dummies(osmdata, prefix_sep="_", columns=cat_columns)
onehotTags = onehotTags.groupby(['osm_id', 'wikidata'], as_index=False).sum(numeric_only=True)

cat_columns = ["prop"]
oneHotWikiProp = pd.get_dummies(wikidataTable, prefix_sep="_", columns=cat_columns)
oneHotWikiProp = oneHotWikiProp.groupby(oneHotWikiProp['wikidata'], as_index=False).sum(numeric_only=True)


cat_columns = ["cls"]
onehotClass = pd.get_dummies(wikidataToConsider, prefix_sep="_", columns=cat_columns)
onehotClass = onehotClass.groupby(onehotClass['wikidata'], as_index=False).sum(numeric_only=True)

tempMerge = pd.merge(oneHotWikiProp, onehotClass, on='wikidata')
Data = pd.merge(onehotTags, tempMerge, on='wikidata' )

#save the data for the particular country and the KG
print('-saving dataset')

Data.to_csv(OUTPUT_FILE, sep='\t', encoding='utf-8', index=False)

with open(QID_INDEX_FILE, 'w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    for l in qid_index:
        writer.writerow(l)
