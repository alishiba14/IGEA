import psycopg2
import csv
from queue import Queue
from threading import Thread
import configparser
import sys


DATA_DIR = sys.argv[1]
CONFIG_PATH = sys.argv[2]

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

PW_FILENAME = config.get('postGIS', 'passwordfile')
PG_HOST = config.get('postGIS', 'host')
PG_USER = config.get('postGIS', 'user')
PG_DB_NAME = config.get('postGIS', 'dbname')
PG_PORT = config.getint('postGIS', 'port')
OUTPUT_FILENAME = DATA_DIR + 'osm rbf.tsv'
TESTRUN = config.getboolean('misc', 'testrun')
LIMIT = config.get('misc', 'limit')
BASE_TABLE = config.get('entity linking', 'base_table')
PREDICTION_TABLE = config.get('entity linking', 'prediction_table')

print('translating osm to rdf')
print('-connecting to database')

with open(PW_FILENAME, 'r', encoding='utf-8') as file:
    password = file.read().strip()

connection = psycopg2.connect(
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

queue = Queue()
stopThreads = False

print('-starting file writing thread')

consumer = Thread(target=consume, daemon=True, args=(lambda: stopThreads, queue, OUTPUT_FILENAME))
consumer.start()

print('-collecting osm entities')

sql = f"""
SELECT g.osm_id, ST_X(ST_Transform(way, 4326)) lon, ST_Y(ST_Transform(way, 4326)) lat, jsonb_strip_nulls(to_jsonb(g)), pe.wkid 
    FROM {BASE_TABLE} g JOIN {PREDICTION_TABLE} pe ON g.osm_id = pe.osm_id
"""

if TESTRUN:
    sql += f" LIMIT {LIMIT}"

cur = connection.cursor()
cur.execute(sql)
for r in cur:
    triplets = []
    id = r[0]
    lon = r[1]
    lat = r[2]

    lat_string = '"'+str(lat)+'"'+'^^<http://www.w3.org/2001/XMLSchema#decimal>'
    lon_string = '"'+str(lon)+'"'+'^^<http://www.w3.org/2001/XMLSchema#decimal>'
    id_string = "<https://www.openstreetmap.org/node/"+str(id)+">"

    triplets.append([id_string, "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                     "<http://www.w3.org/2003/01/geo/wgs84_pos#SpatialThing>"])

    point = '"Point('+str(lat)+' '+str(lon)+')"^^<http://www.opengis.net/ont/geosparql#wktLiteral>'

    triplets.append([id_string, "<http://www.w3.org/2003/01/geo/wgs84_pos#lat>", lat_string])
    triplets.append([id_string, "<http://www.w3.org/2003/01/geo/wgs84_pos#long>", lon_string])
    triplets.append([id_string, "<http://www.w3.org/2003/01/geo/wgs84_pos#Point>", point])

    valid_pairs = []

    for k, v in r[3].items():
        if k not in ['way', 'osm_id', 'tags']:
            # deal with custom columns in db
            valid_pairs.append((k, str(v)))
        elif k == 'tags':
            if v:
                for k_tag, v_tag in v.items():
                    # osm_ tags are only for maintenance and have no real content
                    if not k_tag.startswith('osm_'):
                        valid_pairs.append((k_tag, str(v_tag)))
    for k, v in valid_pairs:
        v = v.replace("\\", "\\\\")
        v = v.replace('"', '\\"')
        v = v.replace('\n', " ")

        k = k.replace(" ", "")

        triplets.append([id_string, "<https://wiki.openstreetmap.org/wiki/Key:"+k+">", '"'+v+'"'])

    for triplet in triplets:
        queue.put(triplet)

stopThreads = True
consumer.join()

connection.close()