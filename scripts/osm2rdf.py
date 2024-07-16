import os
import csv
import sys
import configparser
import asyncio
import asyncpg
from queue import Queue
from threading import Thread

DATA_DIR = sys.argv[1]
CONFIG_PATH = sys.argv[2]

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

PW_FILENAME = config.get('postGIS', 'passwordfile')
PG_HOST = config.get('postGIS', 'host')
PG_USER = config.get('postGIS', 'user')
PG_DB_NAME = config.get('postGIS', 'dbname')
PG_PORT = config.getint('postGIS', 'port')
OUTPUT_FILENAME = os.path.join(DATA_DIR, 'osm_rdf.tsv')
TESTRUN = config.getboolean('misc', 'testrun')
LIMIT = config.getint('misc', 'limit', fallback=1000)
BASE_TABLE = config.get('entity linking', 'base_table')
PREDICTION_TABLE = config.get('entity linking', 'prediction_table')

print('Translating OSM to RDF')

# Read the password
with open(PW_FILENAME, 'r', encoding='utf-8') as file:
    password = file.read().strip()

def consume(stop, queue, filename) -> None:
    """
    Consumer function for threaded file writing
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
                print('- Stopping file writing thread')
                return

queue = Queue()
stop_threads = False

print('- Starting file writing thread')

consumer = Thread(target=consume, daemon=True, args=(lambda: stop_threads, queue, OUTPUT_FILENAME))
consumer.start()

async def fetch_osm_entities():
    conn = await asyncpg.connect(
        user=PG_USER, 
        password=password, 
        database=PG_DB_NAME, 
        host=PG_HOST, 
        port=PG_PORT
    )
    
    sql = f"""
    SELECT g.osm_id, ST_X(ST_Transform(way, 4326)) lon, ST_Y(ST_Transform(way, 4326)) lat, jsonb_strip_nulls(to_jsonb(g)), pe.wkid 
        FROM {BASE_TABLE} g JOIN {PREDICTION_TABLE} pe ON g.osm_id = pe.osm_id
    """
    
    if TESTRUN:
        sql += f" LIMIT {LIMIT}"

    async with conn.transaction():
        rows = await conn.fetch(sql)
        
        for row in rows:
            triplets = []
            id = row['osm_id']
            lon = row['lon']
            lat = row['lat']

            lat_string = f'"{lat}"^^<http://www.w3.org/2001/XMLSchema#decimal>'
            lon_string = f'"{lon}"^^<http://www.w3.org/2001/XMLSchema#decimal>'
            id_string = f"<https://www.openstreetmap.org/node/{id}>"

            triplets.append([id_string, "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                             "<http://www.w3.org/2003/01/geo/wgs84_pos#SpatialThing>"])

            point = f'"Point({lat} {lon})"^^<http://www.opengis.net/ont/geosparql#wktLiteral>'

            triplets.append([id_string, "<http://www.w3.org/2003/01/geo/wgs84_pos#lat>", lat_string])
            triplets.append([id_string, "<http://www.w3.org/2003/01/geo/wgs84_pos#long>", lon_string])
            triplets.append([id_string, "<http://www.w3.org/2003/01/geo/wgs84_pos#Point>", point])

            valid_pairs = []

            for k, v in row['jsonb_strip_nulls'].items():
                if k not in ['way', 'osm_id', 'tags']:
                    valid_pairs.append((k, str(v)))
                elif k == 'tags':
                    if v:
                        for k_tag, v_tag in v.items():
                            if not k_tag.startswith('osm_'):
                                valid_pairs.append((k_tag, str(v_tag)))

            for k, v in valid_pairs:
                v = v.replace("\\", "\\\\")
                v = v.replace('"', '\\"')
                v = v.replace('\n', " ")
                k = k.replace(" ", "")
                triplets.append([id_string, f"<https://wiki.openstreetmap.org/wiki/Key:{k}>", f'"{v}"'])

            for triplet in triplets:
                queue.put(triplet)

    await conn.close()

print('- Collecting OSM entities')
asyncio.run(fetch_osm_entities())

stop_threads = True
consumer.join()

print('- Translation complete')
