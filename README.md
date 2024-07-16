# Iterative Geographic Entity Alignment with Cross-Attention  
This is the README file for the paper Iterative Geographic Entity Alignment with Cross-Attention.
Aim of this file is to explain the steps necessary to run the code provided.
This code has been tested on Python 3.8 and Python 3.9

## SetUp
For easier management of isntalling libraries we suggest using a virtual environment. This can be created in your current working directory with:
```bash
virtualenv venv
```
After activating the virtual environment (e.g. by executing the `activate` script in the venv folder) install the necessary libraries:
```
pip3 install -r requirements.txt
```
Finally download the fasttext model of your choice (e.g. by using the fasttext python library)
```
python3
```
```python
import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')
```
Enter the location of the fasttext model into the config file  

[- import the osm file into postgres -]  
check pbfNodes2postGIS folder for more info

## running the experiment  
The experiment can be run by executing the `runExperiment.py` file
```bash
python3 runExperiment.py
```
If desired, the runExperiment file can be run with an additional argument to specify the config file to be used. If ran without arguments, the `config.ini` file from the `config` folder will be used.  

## config  
The config file contains all configurable information for the experiment. It is structured into parts for the different experiment components. The config contains the following options:

### postGIS  
postGIS contains all information about the database connection.  

| option | use |
| ------ | --- |
| host | hostserver of database |
| user | database user to connect with |
| dbname | name of the database to connect to |
| port | port used for database connection |
| passwordfile | path to text file containing the password for connecting to the database |

### nca  
nca contains information necessary for the schema alignment part of the linking process.

| option | use |
| ------ | --- |
| osm_tag_location | path to a csv file containing wikidata keys matched to osm keys|
|osm_key_location | path to a csv file containing wikidata labels matched to osm tags|
|columns_location | textfile containing the names of all columns into which osm tag information has been moved during migration from osm file to database. **Leave this empty if all Tags are imported into a single tag column** |
|relevance_threshold | minimum amount of entities for a class to be added to the alignment dataset |
|prediction_threshold | minimum confidence for predicting a wikidata osm class match |
|latent_space|dimension of the latent space used in schema alignment|
|num_epochs|number of epochs to train for during schema alignment|
|train_verbose| how much information to display during schema alignment training. Choose from: 0, 1, 2|

### dbpedia scrape  
dbpedia scrape contains information defining how and which information to collect from dbpedia  

| option | use |
| ------ | --- |
|dbpedia_source| dbpedia instance to source data from e.g. fr for `fr.dbpedia.com`|
|country| Country that entities need to be contained in. Usually a uppercase database Relation like `France` check dbpedia for more information|


### wikidata scrape  
wikidata scrape contains information defining which information to scrape from wikidata  

| option | use |
| ------ | --- |
|country_id| Wikidata Qid of the desired country to collect for|
|scrape_values| types of information to scrape. Define as list separated by commas. Example with all possible options: `name, popularity, type labels, full properties`|
|name_language|language to prefer for entity names|

### candidate generation  
candidate generation contains options to adhere to during creation of entity pairs

| option | use |
| ------ | --- |
|method | Method used for candidate generation choose from: distance, name|
|max_candidates| maximum amount of osm candidates to generate per wikidata entry |
|dist_threshold| maximum distance in meters between wikdata entity location and osm entity location to still be considered a possible match |


### fasttext
fasttext contains information about the fasttext model used

| option | use |
| ------ | --- |
|location| path to the fasttext model file to use for encoding |

### entity linking
entity linking contains possible options for entity link prediction

| option | use |
| ------ | --- |
|attention| Use self attention based model|
|attention_dimension| output dimension for self attention|
|linear_dimension| output dimension for linear combination of self attention output|
|epochs|number of epochs to train for|
|train_verbose|how much information to display during entity linking training. Choose from: 0, 1, 2|
|prediction_threshold| confidence threshold to predict two entities to be linked|
|base_table|database table containing osm information (has to be created from osm information previous to the experiment)|
|view_name| name of the view used to store osm information from entities considered for candidate generation (will be created, updated and deleted during the experiment)|
|prediction_table|name for table containing entity match predictions (as well as ground truth marked with iteration = 0)(will be created, updated and deleted during the experiment)|
|index_name|Name of the index created for the temp view|


### meta  
meta contains options for experiment metadata

| option | use |
| ------ | --- |
|data_folder| folder to write experiment files to, auto generated in working directory if left blank|
|num_iterations|number of nca and el iterations to run|
|kg_source| knowledge graph source to use for geo entities. Choose from wikidata or dbpedia.|

### legacy  
Legacy options control different strategies for benchmark runs.

| option | use |
| ------ | --- |
|use_legacy_embeddings| calculate own embeddings based on the osm2kg project. Only a single iteration will be run. |
|num_epochs| number of epochs to train custom embeddings for|
| embedding_dim | dimension of custom embeddings |
|model| **legacy option not used with attention** model to use for prediction, choose from: r_forest, mlp, log_reg, d_tree|
|do_oversampling| **legacy option not used with attention** Flag whether to use SMOTE oversampling during training|

### misc
misc contains other miscellaneous options

| option | use |
| ------ | --- |
|testrun|If set to True, different options during the experiment will be set to reduce runtime and enable a test run for system validation purposes|
|limit|number of entries for dataset to limit to during testruns NOTE: very small datasets can lead to problems such as not finding valid candidates by chance|

## files of interest  
During the experiment information is written into files, that may be of interest for closer inspection and future work.  

| file | content |
| ------ | --- |
| osm rbf.tsv | triplet representation of osm information|
| nca dataset.tsv | matched entities from osm and wikidata as well as encoded information for class match prediction during schema alignment |
| qid_index.tsv | list of wikidata types and their corresponding QIDs |
| predicted classes.tsv | output prediction for matching classes generated during schema alignment |
| create view.sql | sql command used for view generation, containing osm classes that were predicted to match |
| wikidata classes.txt | list of all QIDs of wikidata classes that will be used during entity linking |
| wikidata dump.parquet | parquet file containing all wikidata entities from the given classes |
| pair something | candidate pairs generated for entity linking prediction |
| unmatch something | pairs with no valid match to generate new predicted linked entities |
| el train data.parquet| fasttext embedded training data for entity linking |
| model_report | report entity linking model performance after training |
| model.sav | pickel dump of trained el prediction model |
|predicted matches something| result of model prediction on unmatched pairs |
|keras model| not a file, but the keras model will be saved in this folder for later use (for import instructions look into predictUnmatched)|
|tokenizer.sav|saved tokenizers for attention model. For example implementation look at predictUnmatched|


## scripts  
In this chapter the function of each script will be outlined briefly. The scripts are listed in order of use during the linking pipeline.

| file | task |
| ------ | --- |
|attention|file containing the Attention class for easy availability|
| prepareSchema | generate necessary tables in postgres |
| osm2rdf | generate rdf data of linked osm entities from postgres |
| readRDFWikidata | fetch wikidata information for linked entities generated in osm2rdf |
| schemaMatch | Train model on entity matches an predict class matches |
| reformClasses | use schemaMatch information to generate view in postgres containing osm entities of the predicted class matches, create list of corresponding wikidata qids for entity linking |
| scrapeWikidata | collect all wikidata entities for previous generated QIDs |
| candidateGeneration | generate candidate pairs for wikidata entities from postgres view, split into pairs containing a match and unmatched pairs |
| computeFTEmbeddings | generate fasttext embeddings for entity linking dataset |
| entityLinking | train entity linking predictor |
|entityLinkingAttention| train entity linking predictor with self attention (embeddings will be selfgenerated)|
| predict unmatched | predict possible matches on previously not matchable candidate pairs and write to database |

### legacy Embeddings
Scripts for comparison to previous work
| file | task |
| ------ | --- |
| transformForKV | Use candidate pair table to generate key value pairs for given tags. Key value pairs will be used during generation of embeddings |
| embeddingKeyValue | Use key value pairs to calculate custom embeddings |
| prepareTrainingFromKV | Combine embeddings and training pairs for use in entity Linking. |