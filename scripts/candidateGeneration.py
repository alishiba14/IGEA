import os
import csv
import sys
import configparser
import pandas as pd
import asyncio
import asyncpg
from tqdm import tqdm
from json import dumps
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

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
LOG_FILENAME = os.path.join(DATA_DIR, 'generate_candidates_log.txt')
MATCH_FILENAME = os.path.join(DATA_DIR, 'train_pairs.tsv')
NO_MATCH_FILENAME = os.path.join(DATA_DIR, 'unmatched_pairs.tsv')
DATA_PATH = os.path.join(DATA_DIR, 'wikidata_dump.parquet')
TESTRUN = config.getboolean('misc', 'testrun')
LIMIT = config.getint('misc', 'limit')

with open(PW_FILENAME, 'r') as file:
    password = file.read().strip()

wiki_data = pd.read_parquet(DATA_PATH, engine='pyarrow')

if TESTRUN:
    print(f'Restricting candidate generation to {LIMIT} entities')
    wiki_data = wiki_data[:LIMIT]


def filter_tags_concat(data: dict) -> str:
    tag = []
    for k, v in data.items():
        if k not in ['way', 'osm_id', 'tags']:
            tag.extend([k, v])
        elif k == 'tags':
            if v:
                for k_tag, v_tag in v.items():
                    if k_tag not in ['wikidata', 'wikipedia'] and not k_tag.startswith('osm_'):
                        tag.extend([k_tag, v_tag])
    return ' '.join(tag).replace('\n', ' ')


def filter_tags_json(data: dict) -> str:
    tags_dict = {}
    for k, v in data.items():
        if k not in ['way', 'osm_id', 'tags']:
            tags_dict.update({k: v})
        elif k == 'tags':
            if v:
                for k_tag, v_tag in v.items():
                    if k_tag not in ['wikidata', 'wikipedia'] and not k_tag.startswith('osm_'):
                        tags_dict.update({k_tag: v_tag})
    return dumps(tags_dict)


def consume(stop, queue, filename) -> None:
    with open(filename, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        while True:
            if not queue.empty():
                i = queue.get()
                writer.writerow(i)
            elif stop():
                print('- Stopping file writing thread')
                return


async def fetch_candidates_for_point(conn, wiki_id, location, data, pair_queue, single_queue=None, threshold=2500, limit=100):
    sql = f"""SELECT osm_id, ST_DISTANCE(way, ST_Transform(ST_GeomFromEWKT($1), 3857)) dist, jsonb_strip_nulls(to_jsonb(g)), wkid
              FROM {VIEW_NAME} g
              WHERE ST_DWithin(way, ST_Transform(ST_GeomFromEWKT($2), 3857), $3)
              ORDER BY dist ASC LIMIT $4"""

    is_linked = False
    res = []
    async with conn.transaction():
        async for record in conn.cursor(sql, f'SRID=4326; {location}', f'SRID=4326; {location}', threshold, limit):
            match = False
            id = record['wkid']
            if DATA_SOURCE == 'dbpedia':
                id = id[3:].replace(' ', '_')
            if id == wiki_id:
                is_linked = True
                match = True
            res.append([wiki_id, record['osm_id'], match, record['dist'], filter_tags_concat(record['jsonb_strip_nulls'])] + data)

    if is_linked:
        for entry in res:
            pair_queue.put(entry)
    else:
        for entry in res:
            single_queue.put(entry)


async def fetch_candidates_for_name(conn, wiki_id, name, data, pair_queue, single_queue=None, limit=100):
    sql = f"""SELECT osm_id, (similarity(lower(g."name"), lower($1))) as sim, jsonb_strip_nulls(to_jsonb(g)), wkid
              FROM {VIEW_NAME} g
              WHERE g.name is not null
              ORDER BY sim DESC LIMIT $2"""

    is_linked = False
    res = []
    async with conn.transaction():
        async for record in conn.cursor(sql, name, limit):
            match = False
            id = record['wkid']
            if DATA_SOURCE == 'dbpedia':
                id = id[3:].replace(' ', '_')
            if id == wiki_id:
                is_linked = True
                match = True
            res.append([wiki_id, record['osm_id'], match, record['sim'], filter_tags_concat(record['jsonb_strip_nulls'])] + data)

    if is_linked:
        for entry in res:
            pair_queue.put(entry)
    else:
        for entry in res:
            single_queue.put(entry)


async def fetch_candidates_legacy(conn, wiki_id, location, data, pair_queue, single_queue=None, threshold=2500, limit=100):
    sql = f"""SELECT osm_id, ST_DISTANCE(way, ST_Transform(ST_GeomFromEWKT($1), 3857)) dist, jsonb_strip_nulls(to_jsonb(g)), wkid
              FROM {VIEW_NAME} g
              WHERE ST_DWithin(way, ST_Transform(ST_GeomFromEWKT($2), 3857), $3)
              ORDER BY dist ASC LIMIT $4"""

    is_linked = False
    res = []
    async with conn.transaction():
        async for record in conn.cursor(sql, f'SRID=4326; {location}', f'SRID=4326; {location}', threshold, limit):
            id = record['wkid']
            if DATA_SOURCE == 'dbpedia':
                id = id[3:].replace(' ', '_')
            if id == wiki_id:
                is_linked = True
            res.append([wiki_id, record['osm_id'], is_linked, record['dist'], filter_tags_json(record['jsonb_strip_nulls'])] + data)

    if is_linked:
        for entry in res:
            pair_queue.put(entry)
    else:
        for entry in res:
            single_queue.put(entry)


async def main():
    conn = await asyncpg.connect(
        user=PG_USER, 
        password=password, 
        database=PG_DB_NAME, 
        host=PG_HOST, 
        port=PG_PORT
    )

    with open(LOG_FILENAME, 'w', encoding='utf-8') as file:
        file.write('Starting candidate search\n')
        file.write(f'Matched entities written to: {MATCH_FILENAME}\n')
        file.write(f'Unmatched entities written to: {NO_MATCH_FILENAME}\n')

    match_queue = Queue()
    single_queue = Queue()
    stop_threads = False

    start_time = time.time()

    print('- Starting file writing threads')

    with open(LOG_FILENAME, 'a', encoding='utf-8') as file:
        file.write('Starting consumer threads\n')

    match_consumer = Thread(target=consume, daemon=True, args=(lambda: stop_threads, match_queue, MATCH_FILENAME))
    match_consumer.start()
    single_consumer = Thread(target=consume, daemon=True, args=(lambda: stop_threads, single_queue, NO_MATCH_FILENAME))
    single_consumer.start()

    with open(LOG_FILENAME, 'a', encoding='utf-8') as file:
        file.write('Starting candidate generation threads\n')

    print('- Starting candidate generation')
    print(f'- Method used: {GENERATION_METHOD}')
    if USE_LEGACY_EMBEDDINGS:
        print('- Loading with JSON data for legacy embeddings')

    tasks = []

    if not USE_LEGACY_EMBEDDINGS:
        if GENERATION_METHOD == 'distance':
            col_mask = [c not in ['wkid', 'location'] for c in wiki_data.columns]
            header_names = ['wkid', 'osm_id', 'match', 'dist', 'tags'] + list(wiki_data.columns[col_mask])
            match_queue.put(header_names)
            single_queue.put(header_names)
            for index, row in wiki_data.iterrows():
                tasks.append(fetch_candidates_for_point(conn, row['wkid'], row['location'], list(row[col_mask]), match_queue, single_queue, DIST_THRESHOLD, MAX_CANDIDATES))
        else:  # GENERATION_METHOD == 'name':
            col_mask = [c not in ['wkid', 'name'] for c in wiki_data.columns]
            header_names = ['wkid', 'osm_id', 'match', 'sim', 'tags'] + list(wiki_data.columns[col_mask])
            match_queue.put(header_names)
            single_queue.put(header_names)
            for index, row in wiki_data.iterrows():
                tasks.append(fetch_candidates_for_name(conn, row['wkid'], row['name'], list(row[col_mask]), match_queue, single_queue, MAX_CANDIDATES))
    else:
        col_mask = [c not in ['wkid', 'location', 'name'] for c in wiki_data.columns]
        header_names = ['wkid', 'osm_id', 'match', 'dist', 'tags'] + list(wiki_data.columns[col_mask])
        match_queue.put(header_names)
        single_queue.put(header_names)
        for index, row in wiki_data.iterrows():
            tasks.append(fetch_candidates_legacy(conn, row['wkid'], row['location'], list(row[col_mask]), match_queue, single_queue, DIST_THRESHOLD, MAX_CANDIDATES))

    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc='- Finding candidates'):
        await f

    with open(LOG_FILENAME, 'a', encoding='utf-8') as file:
        file.write('Finished candidate generation threads\n')

    stop_threads = True
    match_consumer.join()
    single_consumer.join()

    with open(LOG_FILENAME, 'a', encoding='utf-8') as file:
        file.write('Stopped consumer threads\n')

    await conn.close()

    with open(LOG_FILENAME, 'a', encoding='utf-8') as file:
        file.write('Closed connection\n')
        file.write(f"Execution ended successfully at {time.strftime('%d.%M.%Y %H:%M:%S', time.gmtime(time.time()))}\n")
        file.write(f"Execution time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}\n")


if __name__ == '__main__':
    asyncio.run(main())
