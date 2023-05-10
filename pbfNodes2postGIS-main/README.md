# pbfNodes2postGIS  

This project is created to support the WorldKG entity linking projects. Its use is to take a current state of a country in osm and write its nodes to a given postGIS database.

The python script will download the osm data and create the configuration file for osm2pgsql.

Make sure to install osm2pgsql and check that it is working. Enable the required extensions (postgis and hstore)   

Install all required python packages. We suggest using a virtual environment.  
```
virtualenv venv
```
```
pip install -r requirements.txt
```

Set up the configuration according to preference. For more see the config chapter.

run the `pbf2postgis.py` script. It will create a command file containing the command to start the database import as well as the profile-lua-script to handle the necessary settings.

Copy the shell command and run it. Enter your database password. The import should start and finish on its own now.  

## config  
The config allows to tweak and enter information for easy downloading and use of the import scripts.  

### postGIS  
Database settings  

| option | use |
| ------ | --- |
| host | database host |
| user | user to log in and create and fill tables with |
| dbname | database name to connect to |
| port | port to connect to |
| target_table | Table to create and fill with node data. Note: we suggest to use underscores instead of spaces to prevent later errors in sql queries|  

### Data  
Settings for pbf data files  

| option | use |
| ------ | --- |
| data_folder | folder to save datafiles to (will be created if not currently present)|

### PBF  
Settings for osm data

| option | use |
| ------ | --- |
| country | Name of the country to get Data for. For inspiration check Geofabrik |