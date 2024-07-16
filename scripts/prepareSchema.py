import sys
import configparser
import asyncio
import asyncpg

CONFIG_PATH = sys.argv[1]

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

PW_FILENAME = config.get('postGIS', 'passwordfile')
PG_HOST = config.get('postGIS', 'host')
PG_USER = config.get('postGIS', 'user')
PG_DB_NAME = config.get('postGIS', 'dbname')
PG_PORT = config.getint('postGIS', 'port')
TABLE_NAME = config.get('entity linking', 'prediction_table')
BASE_TABLE = config.get('entity linking', 'base_table')
VIEW_NAME = config.get('entity linking', 'view_name')
INDEX_NAME = config.get('entity linking', 'index_name')
DATA_SOURCE = config.get('meta', 'kg_source')

print('preparing schema')

with open(PW_FILENAME, 'r') as file:
    password = file.read().strip()

delete_index_sql = f"DROP INDEX IF EXISTS {INDEX_NAME}"
delete_view = f"DROP MATERIALIZED VIEW IF EXISTS {VIEW_NAME}"
delete_table = f"DROP TABLE IF EXISTS {TABLE_NAME}"

sql = f"""
CREATE TABLE {TABLE_NAME}(
	id BIGSERIAL,
	wkid text not null,
	osm_id BIGINT not null,
	confidence float not null,
	iteration int not null
)
"""

if DATA_SOURCE == 'wikidata':
    init_sql = f"""
    INSERT INTO {TABLE_NAME} (wkid, osm_id, confidence, iteration)
        SELECT tags -> 'wikidata', osm_id, 1.0, 0
        FROM {BASE_TABLE}
        WHERE tags -> 'wikidata' is not null
    """
else:
    # DATA_SOURCE == 'dbpedia':
    init_sql = f"""
    INSERT INTO {TABLE_NAME} (wkid, osm_id, confidence, iteration)
        SELECT tags -> 'wikipedia', osm_id, 1.0, 0
        FROM {BASE_TABLE}
        WHERE tags -> 'wikipedia' is not null
    """

async def execute_sql():
    conn = await asyncpg.connect(
        user=PG_USER,
        password=password,
        database=PG_DB_NAME,
        host=PG_HOST,
        port=PG_PORT
    )

    async with conn.transaction():
        cur = conn.cursor()
        print('-deleting old view')
        await conn.execute(delete_index_sql)
        await conn.execute(delete_view)
        print('-deleting old prediction table')
        await conn.execute(delete_table)
        print(f'-creating table {TABLE_NAME}')
        await conn.execute(sql)
        print(f'-filling {TABLE_NAME} with ground truth')
        await conn.execute(init_sql)

    await conn.close()

asyncio.run(execute_sql())
