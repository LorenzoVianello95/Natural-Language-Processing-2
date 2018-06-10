import numpy as np

#GIVEN THE SENTENCES IN THE LEMMAS FORM AND THE DICT THAT MAP LEMMA IN NUMBER RETURN SENTENCES IN NUMB FORM
def build_vectDataset(sentences, voc):
    datal=[]
    for s in sentences:
        datal_s = []
        for w in s:
            if w in voc:
                datal_s.append(voc[w])
            else:
                print("Problems:"+w)
        datal.append(datal_s)
    return datal


#FUNCTION OPEN THE FILES THAT CONTAIN STRING LIKE senseval2.d000.s002.t000  bn:00030861n AND BUILDS DICTIONARIES
def build_idtowsense(file):
    d = {}
    with open(file, 'r') as f:
        for line in f:
            (key, val) = line.split(" ")
            d[key] = val.replace('\n', '')
    return d


#FUNCTION USED TO BUILD THE DICTIONARY THAT MAP THE LABELS IN THE NUMBER SPACE
def build_id_dict(dict1,dict2,index):
    id_dict = {}
    for h in dict1.values():
        if h not in id_dict:
            id_dict[h] = index
            index = index + 1
    for h in dict2.values():
        if h not in id_dict:
            id_dict[h] = index
            index = index + 1
    return id_dict


stoplist = set(["wf"])#,"lemma","pos"
stopwords = set([w.rstrip('\r\n') for w in open("StopwordsM.txt")])

#from xml file extract the sentences, divided by type:lemmas, pos, tokens, labels..
#in the sentences I choose of don't put punctuation and the numbers are replaced by "NUM"
#lemmas: list of list dove ogni sottolista rappresenta una sentence e
#          al suo interno ci sono tutti i lemmas di quella sentence in ordine
#poses: list of list dove ogni sottolista rappresenta una sentence e
#          al suo interno ci sono tutti i poses di quella sentence in ordine
#           inoltre se il lemma corrispondente e' nome, adjettivo , o verbo
#           allora sostituisco direttamente con il suo corrispondente valore all'interno di d_idtws
#tokens: list of list dove ogni sottolista rappresenta una sentence e
#          al suo interno ci sono tutti i tokens di quella sentence in ordine
#maxlen= lunghezza massima delle sentence, mi serve per costruire i placeholder in maniera adeguata
def convert(xml_file,d_idtws):
    lemmas=[]
    poses = []
    tokens = []
    wsd=[]
    wsd_onlyId=[]
    maxlen=0
    with open(xml_file, "r") as f:  # notice the "rb" mode
        lemmas_s = []
        poses_s= []
        tokens_s = []
        wsd_s = []
        wsd_onlyId_s=[]
        for line in f.readlines():
            if len(lemmas)< 50000:
                line = line.replace('>', ' ').replace('<', ' ').replace('"', ' ').replace('=', ' ').replace('/', ' ')
                split = line.strip().split()
                split = [word for word in split if (word not in stoplist) and (len(word) > 0)]
                if len(split)>0:
                    if split[0]=="sentence":
                        if len(lemmas_s)>0:
                            lemmas.append(lemmas_s)
                            poses.append(poses_s)
                            tokens.append(tokens_s)
                            wsd.append(wsd_s)
                            wsd_onlyId.append(wsd_onlyId_s)
                            if len(lemmas_s)>maxlen:
                                #print(lemmas_s)
                                maxlen=len(lemmas_s)
                            lemmas_s = []
                            poses_s = []
                            tokens_s = []
                            wsd_s=[]
                            wsd_onlyId_s=[]
                    elif split[0] == "text" or split[0] == "corpus" or split[0] == "?xml" or split[0] == "det":
                        pass
                    elif len(split)>3 and split[1] == "id":
                        #print(split)
                        poses_s.append(d_idtws[split[2]])
                        wsd_s.append(d_idtws[split[2]])
                        wsd_onlyId_s.append(d_idtws[split[2]])
                        lemmas_s.append(split[4])
                        tokens_s.append(split[7])
                    elif len(split)==5:
                        if split[3]== "NUM":
                            poses_s.append("NUM")
                            lemmas_s.append("NUM")
                            wsd_s.append("NUM")
                            tokens_s.append("NUM")
                            wsd_onlyId_s.append(0)
                        elif split[3]== ".":
                            pass
                        elif split[1] in ['&#178;','&apos;em', '&apos;ll', '&amp;amp;', '&apos;', '&apos;d', '&apos;s', '**f', '**f-value', '-gt']:
                            pass
                        else:
                            if split[1] not in stoplist:
                                poses_s.append(split[3])
                                lemmas_s.append(split[1])
                                wsd_s.append(split[1])
                                tokens_s.append(split[4])
                                wsd_onlyId_s.append(0)
    return lemmas,poses,tokens,maxlen,wsd,wsd_onlyId

#very similar to the precedent but do for the devietion set
#from xml file extract the sentences, divided by type:lemmas, pos, tokens, labels..
#return also a vector that help to divide deviation data in subset like senseval2
#in the sentences I choose of don't put punctuation and the numbers are replaced by "NUM"
def convert_dev(xml_file,d_idtws):
    dataset=[]
    lemmas=[]
    poses = []
    tokens = []
    wsd=[]
    wsd_onlyId=[]
    maxlen=0
    with open(xml_file, "r") as f:  # notice the "rb" mode
        lemmas_s = []
        poses_s= []
        tokens_s = []
        wsd_s = []
        wsd_onlyId_s=[]
        for line in f.readlines():
            line = line.replace('>', ' ').replace('<', ' ').replace('"', ' ').replace('=', ' ').replace('/', ' ')
            split = line.strip().split()
            split = [word for word in split if (word not in stoplist) and (len(word) > 0)]
            if len(split)>0:
                if split[0]=="sentence":
                    for w in split:
                        #print(w)
                        if w.find("senseval2")>=0:
                            dataset.append("senseval2")
                            #print("S2")
                        elif w.find("senseval3") >= 0:
                            dataset.append("senseval3")
                        elif w.find("semeval2007") >= 0:
                            dataset.append("semeval2007")
                        elif w.find("semeval2013") >= 0:
                            dataset.append("semeval2013")
                        elif w.find("semeval2015") >= 0:
                            dataset.append("semeval2015")
                    if len(lemmas_s)>0:
                        lemmas.append(lemmas_s)
                        poses.append(poses_s)
                        tokens.append(tokens_s)
                        wsd.append(wsd_s)
                        wsd_onlyId.append(wsd_onlyId_s)
                        if len(lemmas_s)>maxlen:
                            #print(lemmas_s)
                            maxlen=len(lemmas_s)
                        lemmas_s = []
                        poses_s = []
                        tokens_s = []
                        wsd_s=[]
                        wsd_onlyId_s=[]
                elif split[0] == "text" or split[0] == "corpus" or split[0] == "?xml" or split[0] == "det":
                    pass
                elif len(split)>3 and split[1] == "id":
                    #print(split)
                    poses_s.append(d_idtws[split[2]])
                    wsd_s.append(d_idtws[split[2]])
                    wsd_onlyId_s.append(d_idtws[split[2]])
                    lemmas_s.append(split[4])
                    tokens_s.append(split[7])
                elif len(split)==5:
                    if split[3]== "NUM":
                        poses_s.append("NUM")
                        lemmas_s.append("NUM")
                        wsd_s.append("NUM")
                        tokens_s.append("NUM")
                        wsd_onlyId_s.append(0)
                    elif split[3]== ".":
                        pass
                    elif split[1] in ['&#178;','&apos;em', '&apos;ll', '&amp;amp;', '&apos;', '&apos;d', '&apos;s', '**f', '**f-value', '-gt']:
                        pass
                    else:
                        if split[1] not in stoplist:
                            poses_s.append(split[3])
                            lemmas_s.append(split[1])
                            wsd_s.append(split[1])
                            tokens_s.append(split[4])
                            wsd_onlyId_s.append(0)

    return lemmas,poses,tokens,maxlen,wsd,wsd_onlyId,dataset


#fill the sentences too short using a zero padding
def create_padding(listoflist, max_length):
    cb_lol= []
    for s in listoflist:
        data_s =list(s)
        while len(data_s)<max_length:
            data_s.append(0)
        cb_lol.append(data_s)

    return cb_lol


#method used to split sentences too long, as to reduce the padding dimension and the nn dimension
#the split is made at the half so we mantein an equilibrated sentences
def split_sentences(sentencesX,sentencesY,len_senteces, maxlen):

    for s in sentencesX:
        #print(s)
        if len(s)>maxlen:
            d = [s[i] for i in range(0, round(len(s)/2))]
            e = [s[i] for i in range(round(len(s)/2), len(s))]
            #sentences.remove(s)
            sentencesX.append(e)
            sentencesX.append(d)
    for s in sentencesY:
        #print(s)
        if len(s)>maxlen:
            d = [s[i] for i in range(0, round(len(s)/2))]
            e = [s[i] for i in range(round(len(s)/2), len(s))]
            #sentences.remove(s)
            sentencesY.append(e)
            sentencesY.append(d)

    t_x= [a for a in sentencesX if len(a)<=maxlen]
    t_y = [a for a in sentencesY if len(a) <= maxlen]
    new_len_sentences=[len(i) for i in t_x ]
    return t_x,t_y,new_len_sentences


#method used to divide the deviation set in subsets
def divide_dev(list_to_divide, domains):
    s2=[]
    s3=[]
    s7=[]
    s13=[]
    s15=[]

    for index in range(len(domains)):
        if domains[index] == "senseval2":
            s2.append(list_to_divide[index])
        if domains[index] == "senseval3":
            s3.append(list_to_divide[index])
        if domains[index] == "semeval2007":
            s7.append(list_to_divide[index])
        if domains[index] == "semeval2013":
            s13.append(list_to_divide[index])
        if domains[index] == "semeval2015":
            s15.append(list_to_divide[index])

    return s2,s3,s7,s13,s15




def build_testset(file):
    d = {}
    t=[]
    l=[]
    pos=[]
    id=[]

    with open(file, 'r') as f:
        for line in f:
            l_m=[]
            id_m=[]
            pos_m=[]
            t_m=[]
            split_sent = line.split(" ")
            for w in split_sent:
                split_word = w.split("|")
                if len(split_word)>1:
                    if not(split_word[2]=="."):
                        if split_word[2]=="NUM":
                            t_m.append(split_word[2])
                            l_m.append(split_word[2])
                            pos_m.append(split_word[2])
                        else:
                            t_m.append(split_word[0])
                            l_m.append(split_word[1])
                            pos_m.append(split_word[2])
                        if len(split_word)>3:
                            id_m.append(split_word[3])
                        else:
                            id_m.append(0)
                else:
                    #print(l_m)
                    l.append(l_m)
                    id.append(id_m)
                    pos.append(pos_m)
                    t.append(t_m)

    return l,id,pos,t




