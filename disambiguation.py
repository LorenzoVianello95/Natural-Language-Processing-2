from data_processing import build_idtowsense, convert, build_id_dict, build_testset, build_vectDataset,convert_dev
import numpy as np
import sys
from gensim.models import Word2Vec
import pickle

#IN THIS FILE ARE PREPARED ALL THE PREPROCESSED DATA

# BUILD DICTIONARIES THAT LINK senseval2.d000.s002.t000  TO  bn:00030861n (for example)
d_idtws_train=build_idtowsense('TRAIN/semcor.gold.key.bnids.txt')
d_idtws_dev= build_idtowsense('DevData/ALL.gold.key.bnids.txt')

#BUILD LIST OF LIST WHERE EACH SUBLIST IS A SENTENCE,
# IN LIST l ARE PRESENT ALL SENTENCES RAPPRESENTED USING LEMMA [[the,cat, be,..],[I,like,icecream..]...]
#IN LIST p ARE PRESENT ALL SENTENCES RAPPRESENTED USING PART OF SPEECH [[NOUN,VERB...]]
#IN LIST t ARE THE TOKEN FORMS
#IN THE LIST Y THERE ARE THE "LABELS" [[I,bn:00030861v,in,....]]   SO BOTH LEMMA AND BABELNET SENSES
#I DO THE SAME THING FOR TRAIN, DEV AND TEST
l,p,t,MAX_LENGTH,y,y_onlyId = convert('TRAIN/semcor.data.xml',d_idtws_train)
l_dev,p_dev,t_dev,MAX_LENGTH_dev,y_dev,y_onlyId_dev,datasets_dev = convert_dev('DevData/ALL.data.xml',d_idtws_dev)
l_test, id_test, _,_ = build_testset('TEST/test_data.txt')

NUM_SENTECES=len(p)

#BUILD THE EMBEDDING MODEL USING THE GENSIM W2V BASED ON MY DATA
model = Word2Vec(l+l_dev+l_test, size=300 ,min_count=1) #costruisco l'embedding usando gensim
#model = Word2Vec(l+l_dev, min_count=1)
#filename = 'GoogleNews-vectors-negative300.bin'
#model = KeyedVectors.load_word2vec_format(filename, binary=True)
print(model)

#BUILD A DICTIONARY THAT LINK A WORD TO A NUMBER, WHERE THE NUMBER IS THE ROW OF THE EMBEDDING MATRIX
words = list(model.wv.vocab)
words=sorted(words) # riordinate in ordine alfabetico #QUESTO DI FATTO È IL DIZIONARIO DEGLI INPUT
words_voc={k: words.index(k) for k in words }

#BUILD A DICTIONARY THAT LINK A BABEL SENSE TO A NUMBER, WHERE THE NUMBER IS BIGGER THEN THE DIMENSION OF THE PRECEDENT DIC
id_dict=build_id_dict(d_idtws_dev,d_idtws_train,len(words_voc))
#BUILD EMBEDDING MATRIX
embeddings=np.array([model[w] for w in words])

#TRANSFORM THE LEMMAS'S FORM SENTENCES IN NUMBER'S FORM
list_with_embedding= build_vectDataset(l,words_voc)
list_with_embedding_dev= build_vectDataset(l_dev,words_voc)
list_with_embedding_test= build_vectDataset(l_test,words_voc)

#VECTORS THAT CONTAINS ALL THE LENGTH OF THE SENTENCES
lentgh_of_sentences=[len(s) for s in l]
lentgh_of_sentences_dev=[len(s) for s in l_dev]

#UNIFY THE 2 VOCABULARY CREATED, THE ONE WORD-NUMB AND THE ONE BABELSENSE-NUMB
words_voc.update(id_dict)
final_voc=words_voc

#TRANSFORM THE (LEMMAS,BABELSENS)'S FORM SENTENCES IN NUMBER'S FORM
y_pos= build_vectDataset(y,final_voc)
y_pos_dev= build_vectDataset(y_dev,final_voc)

#NUMBER OF POSSIBLE TAGS
ntags=len(final_voc)

#THOSE ARE THE INPUTS OF MY NN
X=list_with_embedding
Y=y_pos
X_dev=list_with_embedding_dev #al momento questi non sono ancora divisi
Y_dev=y_pos_dev

#SAVE THEM IN FILES AS TO READ FASTER
f = open('embedding.pckl', 'wb')
pickle.dump(embeddings, f)
f.close()
f = open('X.pckl', 'wb')
pickle.dump(X, f)
f.close()
f = open('Y.pckl', 'wb')
pickle.dump(Y, f)
f.close()
f = open('lentgh_of_sentences.pckl', 'wb')
pickle.dump(lentgh_of_sentences, f)
f.close()
f = open('array_of_elements.pckl', 'wb')
pickle.dump([NUM_SENTECES,ntags], f)
f.close()
#Saving deviation variables
f = open('X_dev.pckl', 'wb')
pickle.dump(X_dev, f)
f.close()
f = open('Y_dev.pckl', 'wb')
pickle.dump(Y_dev, f)
f.close()
f = open('lentgh_of_sentences_dev.pckl', 'wb')
pickle.dump(lentgh_of_sentences_dev, f)
f.close()
f = open('datasets_dev.pckl', 'wb')
pickle.dump(datasets_dev, f)
f.close()
f = open('final_voc.pckl', 'wb')
pickle.dump(final_voc, f)
f.close()


print("Preprocessing concluso")
