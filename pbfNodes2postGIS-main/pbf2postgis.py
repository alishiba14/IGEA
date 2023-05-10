import pyrosm
import configparser
import sys
import os

if len(sys.argv) >= 2:
    CONFIG_PATH = sys.argv[1]
else:
    CONFIG_PATH = 'config.ini'

config = configparser.ConfigParser()
config.read(CONFIG_PATH)


COUNTRY = config.get('pbf', 'country')

PG_HOST = config.get('postGIS', 'host')
PG_USER = config.get('postGIS', 'user')
PG_DB_NAME = config.get('postGIS', 'dbname')
PG_PORT = config.getint('postGIS', 'port')
PG_TABLE = config.get('postGIS', 'target_table')

DATA_DIR = config.get('data', 'data_folder')

os.makedirs(DATA_DIR, exist_ok=True)

file_path = pyrosm.get_data(COUNTRY, directory=DATA_DIR, update=False)

lua_script = """
local srid = 3857

local tables = {}

tables.node = osm2pgsql.define_table{
	name = "%s",
	ids = { type = 'node', id_column = 'osm_id' },
	columns = {
		{ column = 'tags', type = 'hstore' },
		{ column = 'way', type = 'point', not_null = true },
	}
}

function clean_tags(tags)
    tags.odbl = nil
    tags.created_by = nil
    tags.source = nil
    tags['source:ref'] = nil

    return next(tags) == nil
end

function osm2pgsql.process_node(object)

    if clean_tags(object.tags) then
        return
    end

    
	tables.node:insert({
		tags = object.tags,
		way = object:as_point()
	})
end
""" % PG_TABLE

lua_file = f'{str(COUNTRY).lower()} nodes.lua'

with open(lua_file, 'w', newline='') as file:
    file.write(lua_script)

shell_command = f"osm2pgsql -U {PG_USER} -W -d {PG_DB_NAME} -H {PG_HOST}  -P {PG_PORT} -O flex -S \"{lua_file}\" \"{file_path}\""

shell_file = f'{str(COUNTRY).lower()} command.txt'

with open(shell_file, 'w') as file:
    file.write(shell_command)
