
# coding: utf-8


import random
import json
import os
from bs4 import BeautifulSoup
import re
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, wordpunct_tokenize, pos_tag
from nltk.corpus import stopwords
import nltk
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim import corpora, models, similarities
from gensim.models.phrases import Phraser, Phrases
from gensim.corpora import Dictionary
from sklearn.cluster import MiniBatchKMeans
import numpy as np



def paras(html):
    soup = BeautifulSoup(html, 'html.parser')
    for string in soup.stripped_strings:
        yield string

def sents(html):
    for paragraph in paras(html):
        for sentence in sent_tokenize(paragraph):
            yield sentence

def words(html):
    for sentence in sents(html):
        words = nltk.wordpunct_tokenize(sentence)
        words=[word.lower() for word in words if word.isalpha()]
        for token in words:
            yield token


def tokenize(html):
    for paragraph in paras(html):
        yield [pos_tag(wordpunct_tokenize(sent)) for sent in sent_tokenize(paragraph)]





docs = []
titles = []
stop_words = set(stopwords.words('english'))
wnl = WordNetLemmatizer()
n=0
path = 'corpus_comb' + '/'
files = os.listdir(path)
rxs = [re.search('^\.',file) for file in files]
i =[n for n in range(len(rxs)) if rxs[n]]
m = 0
for k in i:
    files.pop(k-m)
    m += 1

for file in files: 
    with open(path + file, encoding='utf-8') as f:
        job_post = f.read()

    job_dict = eval(job_post)

    job_dict.keys()

    title = job_dict['job title']
    company = job_dict['company']
    titles.append([n, title, company])
    html = job_dict['job description']
    doc = []
    xs = words(html)
    for x in xs:
        if x not in stop_words:
            doc.append(x.lower())

    docs.append(doc)
    n += 1




corpus = [d for d in docs]


phrases = Phrases(docs, min_count=1, threshold=1)

bigram_transformer = Phraser(phrases)

n_vectors = 1000

documents = [TaggedDocument(bigram_transformer[doc], [i]) for i, doc in enumerate(docs)]
model = Doc2Vec(documents, vector_size=n_vectors, dm=1, window=10, min_count=1, epochs=50, ns_exponent=0.5, dbow_words=1)


stop_words = set(stopwords.words('english'))
wnl = WordNetLemmatizer()
n=0
resses = []

with open('/Users/grothjd/documents/DS_resume/Jonathan_Groth_PhD_Resume_test.txt', encoding='utf-8') as f:
    resume = f.read()

xs = words(resume)
for x in xs:
    if x not in stop_words:
        resses.append(wnl.lemmatize(x).lower())



inferred_vector = model.infer_vector(bigram_transformer[resses])
sims = model.docvecs.most_similar([inferred_vector])

select = []
for sim in sims:
    select.append(sim[0])




job = []
for title in titles:
    if int(title[0]) in select:
        job.append(title)
print(job)




keywords = model.wv.most_similar(["programming","language"], topn=100)




#print(keywords)


word2vec_dict={}
for i in model.wv.vocab.keys():
    try:
        word2vec_dict[i]=model[i]
    except:
        pass

clusters = MiniBatchKMeans(n_clusters=50, max_iter=10,batch_size=200,
                        n_init=1,init_size=2000)
X = np.array([value.T for key, value in word2vec_dict.items()])
y = [key for key, value in word2vec_dict.items()]
clusters.fit(X)
from collections import defaultdict
cluster_dict=defaultdict(list)
for word,label in zip(y,clusters.labels_):
    cluster_dict[label].append(word)


#for i in range(len(cluster_dict)):
#    if 'domain' in cluster_dict[i]:
#        cluster_dict[i].sort()
#        print(cluster_dict[i])

from training_sets import skills

negative = random.sample(model.wv.vocab.keys(), 600)
neg_list = []
for w in negative:
    if w not in skills:
        neg_list.append(w)

neg_vectors = model[neg_list[0]]
print(neg_vectors.shape)
for i in range(1,len(neg_list)):
    vec = model[neg_list[i]]
    neg_vectors = np.concatenate((neg_vectors, vec))

pos_vectors = model[skills[0]]
for i in range(1,len(skills)):
    try:
        vec = model[skills[i]]
        pos_vectors = np.concatenate((pos_vectors, vec))
    except Exception as e:
        print(e)

n_neg = int(neg_vectors.shape[0]/n_vectors)
n_pos = int(pos_vectors.shape[0]/n_vectors)
vectors_neg = np.reshape(neg_vectors, (n_neg,n_vectors))
vectors_pos = np.reshape(pos_vectors, (n_pos,n_vectors))

vectors_array = np.concatenate((vectors_neg, vectors_pos),0)

labels_neg = np.zeros(n_neg)
labels_pos = np.ones(n_pos)

labels_array = np.concatenate((labels_neg, labels_pos))


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(vectors_array, labels_array, test_size=0.30, random_state=2)

print(y_test)

model_clf = SVC(C=5, kernel='rbf', gamma=0.1)
model_clf.fit(X_train, y_train)

model_RF = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=43)
model_RF.fit(X_train, y_train)

score = model_clf.score(X_test, y_test)
print("SVC score= " + str(score))

score_rf = model_RF.score(X_test, y_test)
print("Random Forest score= " + str(score_rf))

bigram_doc = bigram_transformer[docs[random.randint(0,1218)]]
resses_vectors = model.wv.get_vector(bigram_doc[0])
print(neg_vectors.shape)
for i in range(1,len(bigram_doc)):
    try:
        vec = model.wv.get_vector(bigram_doc[i])
    except:
        pass
    resses_vectors = np.concatenate((resses_vectors, vec))
n_resses = int(resses_vectors.shape[0]/n_vectors)
vectors_resses = np.reshape(resses_vectors, (n_resses,n_vectors))


pred_out = model_clf.predict(vectors_resses)

pred_skills = []
for i in range(len(pred_out)):
    if pred_out[i] == 1:
        pred_skills.append(bigram_doc[i])

print("---------SVC Predictions-----------")
print(pred_skills)
print("-----------------------------------")

pred_out_rf = model_clf.predict(vectors_resses)

pred_skills_rf = []
for i in range(len(pred_out_rf)):
    if pred_out_rf[i] == 1:
        pred_skills_rf.append(bigram_doc[i])

print("--------Random Forest--------------")
print(pred_skills)
print("-----------------------------------")

print(bigram_doc)
