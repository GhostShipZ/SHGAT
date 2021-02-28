# SHGAT

## Required packages
The code has been tested running under Python 3.6.12, with the following packages installed (along with their dependencies):

- torch==1.4.0
- numpy==1.18.5
- tensorflow-gpu==2.1.0
- scikit-learn==0.24.1
- gensim==3.8.3
- pandas==1.1.4

## Files in the folder
- `data/`
  - `movie/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
  - `music/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
    - `user_artists.dat`: raw rating file of Last.FM;
- `src/`: implementations of SHGAT.

## Run the code
- Movie  
  (The raw rating file of MovieLens-20M is too large to be contained in this repository.
  Download the dataset first.)
  ```
  $ wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
  $ unzip ml-20m.zip
  $ mv ml-20m/ratings.csv data/movie/
  $ cd src
  $ python preprocess.py -d movie-20M
  ```

- Music
  - ```
    $ cd src
    $ python preprocess.py -d music
    ```
  - open `src/main.py` file;
    
  - comment the code blocks of parameter settings for MovieLens-20M;
    
  - uncomment the code blocks of parameter settings for Last.FM;
    
  - ```
    $ python main.py
    ```

