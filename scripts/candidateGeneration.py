from tqdm import tqdm
from json import dumps
import time
from queue import Queue
from threading import Thread
import csv
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from psycopg2.pool import ThreadedConnectionPool
import sys
import configparser
import copy

DATA_DIR = sys.argv[1]
CONFIG_PATH = sys.argv[2]

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

PW_FILENAME = config.get('postGIS', 'passwordfile')
PG_HOST = config.get('postGIS', 'host')
PG_USER = config.get('postGIS', 'user')
PG_DB_NAME = config.get('postGIS', 'dbname')
PG_PORT = config.getint('postGIS', 'port')
VIEW_NAME = config.get('entity linking', 'view_name')
DATA_SOURCE = config.get('meta', 'kg_source')
MAX_CANDIDATES = config.getint('candidate generation', 'max_candidates')
DIST_THRESHOLD = config.getint('candidate generation', 'dist_threshold')
GENERATION_METHOD = config.get('candidate generation', 'method')
USE_LEGACY_EMBEDDINGS = config.getboolean('legacy', 'use_legacy_embeddings')
LOG_FILENAME = DATA_DIR + 'generate candidates log.txt'  # file to write logging metadata to
MATCH_FILENAME = DATA_DIR + 'train pairs.tsv'  # file to write candidate pairs to
NO_MATCH_FILENAME = DATA_DIR + 'unmatched pairs.tsv'  # file to write unmatched wikidata entries to
DATA_PATH = DATA_DIR + 'wikidata dump.parquet'
TESTRUN = config.getboolean('misc', 'testrun')
LIMIT = config.getint('misc', 'limit')

with open(PW_FILENAME, 'r') as file:
    password = file.read().strip()

wiki_data = pd.read_parquet(DATA_PATH, engine='pyarrow')

if TESTRUN:
    print(f'restricting candidate generation to {LIMIT} entities')
    wiki_data = wiki_data[:LIMIT]


def filter_tags_concat(data: dict) -> str:
    """
    Function for data dictionary to filter out meaningless information
    note: keys and values are concatenated and that information is lost, as mean pooling will be used later
    for different embeddings this would need to change
    :param data: dictionary, postgres row in json format
    :return: string containing relevant tags and values
    """
    tag = []
    for k, v in data.items():
        if k not in ['way', 'osm_id', 'tags']:
            tag.extend([k, v])
        elif k == 'tags':
            if v:
                for k_tag, v_tag in v.items():
                    # we don't want wikidata tag to skew the learning
                    # osm_ tags are only for maintenance and have no real content
                    if k_tag not in ['wikidata', 'wikipedia'] and not k_tag.startswith('osm_'):
                        tag.extend([k_tag, v_tag])
    return ' '.join(tag).replace('\n', ' ')

def filter_tags_json(data: dict) -> str:
    """
    Function for data dictionary to filter out meaningless information
    dict will be written to string via json dump
    :param data: dictionary, postgres row in json format
    :return: string containing relevant tags and values
    """
    tags_dict = {}
    tag = []
    for k, v in data.items():
        if k not in ['way', 'osm_id', 'tags']:
            tags_dict.update({k: v})
        elif k == 'tags':
            if v:
                for k_tag, v_tag in v.items():
                    # we don't want wikidata tag to skew the learning
                    # osm_ tags are only for maintenance and have no real content
                    if k_tag not in ['wikidata', 'wikipedia'] and not k_tag.startswith('osm_'):
                        tags_dict.update({k_tag: v_tag})
    return dumps(tags_dict)


tcp = ThreadedConnectionPool(
    minconn=1,
    maxconn=64,
    dbname=PG_DB_NAME,
    user=PG_USER,
    password=password,
    host=PG_HOST,
    port=PG_PORT
)


def consume(stop, queue, filename) -> None:
    """
    consumer function for threaded file writing
    :param stop: function to define stop condition
    :param queue: queue to read content from
    :param filename: filename to write to
    :return:
    """
    with open(filename, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        while True:
            if not queue.empty():
                i = queue.get()
                writer.writerow(i)
            elif stop():
                print('-stopping file writing thread')
                return


def get_candidates_for_point(wiki_id: str,
                             location: str,
                             data: list,
                             t_pool: ThreadedConnectionPool,
                             pair_queue: Queue,
                             single_queue: Queue = None,
                             threshhold: int = 2500,
                             limit: int = 100) -> None:
    """
    function to retrieve possible candidates according to locational distance
    wiki_id: qid of wikidata entity to generate candidate for
    location: Point(lon lat) position of entity
    data: list of information to pass for dataset creation
    t_pool: threaded connection pool for postgres connection
    pair_queue: queue to push valid train pairs to for writing
    single_queue: queue to push unmatched wikidata entries to
    threshhold: maximal distance to still consider database entry as candidate
    limit: maximum number of candidates to generate
    """
    conn = t_pool.getconn()
    cur = conn.cursor()

    sql = f"""SELECT osm_id, ST_DISTANCE(way, ST_Transform(ST_GeomFromEWKT(%s), 3857)) dist, jsonb_strip_nulls(to_jsonb(g)), wkid
    FROM {VIEW_NAME} g
    WHERE ST_DWithin(way, ST_Transform(ST_GeomFromEWKT(%s), 3857), %s)
    ORDER BY dist ASC LIMIT %s
    """

    cur.execute(sql, [f'SRID=4326; {location}', f'SRID=4326; {location}', threshhold, limit])
    is_linked = False  # Flag if any candidate is linked to entity
    res = []
    for r in cur:
        match = False  # flag if this candidate is linked to entity
        if r[3]:
            id = r[3]
            if DATA_SOURCE == 'dbpedia':
                id = id[3:].replace(' ', '_')
            if id == wiki_id:  # only keep candidate set if any candidate is correct for entity
                is_linked = True
                match = True
        res.append([wiki_id, r[0], match, r[1], filter_tags_concat(r[2])] + data)

    if is_linked:
        for entry in res:
            pair_queue.put(entry)
    else:
        for entry in res:
            single_queue.put(entry)
    t_pool.putconn(conn)


def get_candidates_for_name(wiki_id: str,
                           name: str,
                           data: list,
                           t_pool: ThreadedConnectionPool,
                           pair_queue: Queue,
                           single_queue: Queue = None,
                           limit: int = 100) -> None:
    """
    function to retrieve possible candidates according name similarity
    wiki_id: qid of wikidata entity to generate candidate for
    name: wikidata name to match against
    data: list of information to pass through for dataset writing
    t_pool: threaded connection pool for postgres connection
    pair_queue: queue to push valid train pairs to for writing
    single_queue: queue to push unmatched wikidata entries to
    limit: maximum number of candidates to generate
    """
    conn = t_pool.getconn()
    cur = conn.cursor()

    sql = f"""SELECT osm_id, (similarity(lower(g."name") ,lower(%s))) as sim, jsonb_strip_nulls(to_jsonb(g)), wkid
    FROM {VIEW_NAME} g
    WHERE g.name is not null
    ORDER BY sim DESC LIMIT %s"""

    cur.execute(sql, [name, limit])
    is_linked = False  # Flag if any candidate is linked to entity
    res = []
    for r in cur:
        match = False  # flag if this candidate is linked to entity
        if r[3]:
            id = r[3]
            if DATA_SOURCE == 'dbpedia':
                id = id[3:].replace(' ', '_')
            if id == wiki_id:  # only keep candidate set if any candidate is correct for entity
                is_linked = True
                match = True
        res.append([wiki_id, r[0], match, r[1], filter_tags_concat(r[2])] + data)

    if is_linked:
        for entry in res:
            pair_queue.put(entry)
    else:
        for entry in res:
            single_queue.put(entry)
    t_pool.putconn(conn)

def get_candidates_legacy(wiki_id: str,
                          location: str,
                          data: list,
                          t_pool: ThreadedConnectionPool,
                          pair_queue: Queue,
                          single_queue: Queue = None,
                          threshhold: int = 2500,
                          limit: int = 100) -> None:
    """
    function to retrieve possible candidates according to locational distance and return json data for custom embeddings
    wiki_id: qid of wikidata entity to generate candidate for
    location: Point(lon lat) position of entity
    data: list of information to pass for dataset creation
    t_pool: threaded connection pool for postgres connection
    pair_queue: queue to push valid train pairs to for writing
    single_queue: queue to push unmatched wikidata entries to
    threshhold: maximal distance to still consider database entry as candidate
    limit: maximum number of candidates to generate
    """
    conn = t_pool.getconn()
    cur = conn.cursor()

    sql = f"""SELECT osm_id, ST_DISTANCE(way, ST_Transform(ST_GeomFromEWKT(%s), 3857)) dist, jsonb_strip_nulls(to_jsonb(g)), wkid
    FROM {VIEW_NAME} g
    WHERE ST_DWithin(way, ST_Transform(ST_GeomFromEWKT(%s), 3857), %s)
    ORDER BY dist ASC LIMIT %s
    """

    cur.execute(sql, [f'SRID=4326; {location}', f'SRID=4326; {location}', threshhold, limit])
    is_linked = False  # Flag if any candidate is linked to entity
    res = []
    for r in cur:
        if r[3]:
            id = r[3]
            if DATA_SOURCE == 'dbpedia':
                id = id[3:].replace(' ', '_')
            if id == wiki_id:  # only keep candidate set if any candidate is correct for entity
                is_linked = True
                match = True
        res.append([wiki_id, r[0], match, r[1], filter_tags_json(r[2])] + data)

    if is_linked:
        for entry in res:
            pair_queue.put(entry)
    else:
        for entry in res:
            single_queue.put(entry)
    t_pool.putconn(conn)

with open(LOG_FILENAME, 'w', encoding='utf-8') as file:
    file.write('starting candidate search\n')
    file.write(f'matched entities written to: {MATCH_FILENAME}\n')
    file.write(f'unmatched entities written to: {NO_MATCH_FILENAME}\n')

match_queue = Queue()
single_queue = Queue()
stopThreads = False

start_time = time.time()

print('-starting file writing threads')

with open(LOG_FILENAME, 'a', encoding='utf-8') as file:
    file.write('starting consumer threads\n')

# consumer for writing training data
match_consumer = Thread(target=consume, daemon=True, args=(lambda: stopThreads, match_queue, MATCH_FILENAME))
match_consumer.start()
# consumer for writing unmatched wikidata entries
single_consumer = Thread(target=consume, daemon=True, args=(lambda: stopThreads, single_queue, NO_MATCH_FILENAME))
single_consumer.start()

with open(LOG_FILENAME, 'a', encoding='utf-8') as file:
    file.write('starting candidate generation threads\n')

print('-starting candidate generation')
print(f'-method used: {GENERATION_METHOD}')
if USE_LEGACY_EMBEDDINGS:
    print('-loading with json data for legacy embeddings')


with ThreadPoolExecutor(max_workers=64) as pool:
    futures_list = []
    if not USE_LEGACY_EMBEDDINGS:
        if GENERATION_METHOD == 'distance':
            col_mask = [c not in ['wkid', 'location'] for c in wiki_data.columns]
            header_names = ['wkid', 'osm_id', 'match', 'dist', 'tags'] + list(wiki_data.columns[col_mask])
            match_queue.put(header_names)
            single_queue.put(header_names)
            for index, row in wiki_data.iterrows():
                futures_list += [pool.submit(get_candidates_for_point,
                                             row['wkid'],
                                             row['location'],
                                             list(row[col_mask]),
                                             tcp,
                                             match_queue,
                                             single_queue,
                                             2500,
                                             MAX_CANDIDATES)]
        else:# GENERATION_METHOD == 'name':  # chose name similarity
            col_mask = [c not in ['wkid', 'name'] for c in wiki_data.columns]
            header_names = ['wkid', 'osm_id', 'match', 'sim', 'tags'] + list(wiki_data.columns[col_mask])
            match_queue.put(header_names)
            single_queue.put(header_names)
            for index, row in wiki_data.iterrows():
                futures_list += [pool.submit(get_candidates_for_name,
                                             row['wkid'],
                                             row['name'],
                                             list(row[col_mask]),
                                             tcp,
                                             match_queue,
                                             single_queue,
                                             MAX_CANDIDATES)]
    else:
        col_mask = [c not in ['wkid', 'location', 'name'] for c in wiki_data.columns]
        header_names = ['wkid', 'osm_id', 'match', 'dist', 'tags'] + list(wiki_data.columns[col_mask])
        match_queue.put(header_names)
        single_queue.put(header_names)
        for index, row in wiki_data.iterrows():
            futures_list += [pool.submit(get_candidates_legacy,
                                         row['wkid'],
                                         row['location'],
                                         list(row[col_mask]),
                                         tcp,
                                         match_queue,
                                         single_queue,
                                         2500,
                                         MAX_CANDIDATES)]

    # generate progressbar for threaded tasks
    for f in tqdm(as_completed(futures_list), total=len(futures_list), desc='-finding candidates'):
        _ = f.result()

with open(LOG_FILENAME, 'a', encoding='utf-8') as file:
    file.write('finished candidate generation threads\n')

stopThreads = True
match_consumer.join()
single_consumer.join()

with open(LOG_FILENAME, 'a', encoding='utf-8') as file:
    file.write('stopped consumer threads\n')

tcp.closeall()

with open(LOG_FILENAME, 'a', encoding='utf-8') as file:
    file.write('closed connection\n')

with open(LOG_FILENAME, 'a', encoding='utf-8') as file:
    file.write(f"execution ended successfully at {time.strftime('%d.%M.%Y %H:%M:%S', time.gmtime(time.time()))}\n")
    file.write(f"Execution time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
