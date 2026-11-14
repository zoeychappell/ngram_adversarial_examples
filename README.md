# ngram_adversarial_examples
CSEC 720 Deep Learning Security

# Download Database
1. Navigate to https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
2. Select Download -> Download dataset as zip. 
3. Place in directory
4. Unzip and rename the .csv to "IMDB Dataset.csv"

# How to run
1. Train the model with bilstm.py 
2. Tune the model with bilstm_tuning.py
3. Run read_files.py 
4. Run bigram.py to generate bigram candidates. 
5. To recalculate HowNet candidate set, run 
    build_embeddings.py, 
    gen_pos_tag.py, 
    lemma.py, 
    gen_candidates.py
6. Run BU_SPO_fool.py to generate adversarial examples