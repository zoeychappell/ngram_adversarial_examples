'''
Zoey Chappell and Jonah Whitney
Based on Yang et al. 
Completed the necessary changes. 
NOTE: I changed the Part of Speech (POS) tagger entirely 
to use NLTK PosTagger. There was too much extra work to get the 
original Standford POS tagger to work. 
'''
import pickle
import nltk
'''
with open('./all_seqs.pkl', 'rb') as fh:
    train,valid,test = pickle.load(fh)
with open('./nli_tokenizer.pkl', 'rb') as fh:
    tokenizer = pickle.load(fh)
dict = {w: i for (w, i) in tokenizer.word_index.items()}
inv_dict = {i: w for (w, i) in dict.items()}
word_candidate={}
trains=[t[1:-1] for t in train['s2']]
'''
with open('aux_files/dataset_50000.pkl','rb') as fp:
    print(f'Loading dataset from aux_files/dataset_50000.pkl')
    dataset=pickle.load(fp)

nltk.download('averaged_perceptron_tagger_eng')

class PosTagger: 
    def tag(self, tokens):
        return nltk.pos_tag(tokens)

    def sents(self, sentence):
        return nltk.pos_tag_sents([sentence])

pos_tagger = PosTagger()


#jar = 'hownet_candidates/stanford-postagger-2018-10-16/stanford-postagger-2018-10-16/stanford-postagger-3.9.2.jar'
#model = 'hownet_candidates/stanford-postagger-2018-10-16/stanford-postagger-2018-10-16/models/english-left3words-distsim.tagger'
#pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')

#from nltk.stem import WordNetLemmatizer
#wnl = WordNetLemmatizer()
#pos_tag=pos_tagger.tag(['what', "'s", 'invigorating', 'about', 'it', 'is', 'that', 'it', 'does', "n't", 'give', 'a', 'damn'])
#print(pos_tag)
print('Building POS tags for training and test data')
train_text=[[dataset.inv_full_dict[t] for t in tt] for tt in dataset.train_seqs]
test_text=[[dataset.inv_full_dict[t] for t in tt] for tt in dataset.test_seqs]
all_pos_tags=[]
test_pos_tags=[]
print('On for text in train_text')
for text in train_text:
    pos_tags = pos_tagger.sents(text)
    all_pos_tags.append(pos_tags)
print('On for text in test_text')
for text in test_text:
    pos_tags = pos_tagger.sents(text)
    test_pos_tags.append(pos_tags)
print('Saving POS tags to pos_tags.pkl and pos_tags_test.pkl')
f=open('pos_tags.pkl','wb')
pickle.dump(all_pos_tags,f)
f=open('pos_tags_test.pkl','wb')
pickle.dump(test_pos_tags,f)
