from typing import Dict , Text , List
from itertools import chain
import io
import spacy
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
# from sklearn.cross_validation import cross_val_score
# from sklearn.grid_search import RandomizedSearchCV
from collections import defaultdict
import pickle
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

import sys
import os
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../')

from ..utils.TrainDataClass import TrainDataClass

class CRFsuiteEntityExtractorClass():
    def __init__(self):
        self.defaults ={
            "algorithm":"lbfgs",
            "c1":0.1,
            "c2":0.1,
            "max_iterations":100,
            "all_possible_transitions":True
        }
        
        self.nlp = spacy.load('en')
    
    
    
    def _train(self, data: TrainDataClass)->None:
        
        train_sents = self._preproccess_training_data(data)
        
        X_train = [self.sent2features(s) for s in train_sents]
        y_train = [self.sent2labels(s) for s in train_sents]

        self.model = sklearn_crfsuite.CRF(
            algorithm=self.defaults["algorithm"], 
            c1=self.defaults["c1"], 
            c2=self.defaults["c2"], 
            max_iterations=self.defaults["max_iterations"], 
            all_possible_transitions=self.defaults["all_possible_transitions"]
        )
        self.model.fit(X_train, y_train)
    
    def _predict(self ,text: Text)->Dict: 
        sent = self._tokenize_and_tag_text(text)
        features = self.sent2features(sent)
        y_pred = self.model.predict([features])

        return self._labels_to_json(text , y_pred[0])
    
    def _get_labels(self)->List:
        labels = list(crf.classes_)
        labels.remove('O')
        return labels
    
    def _save(self, file_name: Text="entity_model", model_dir: Text="entity_model"):
        
        checkpoint = os.path.join(model_dir, file_name + ".ckpt")

        try:
            os.makedirs(os.path.dirname(checkpoint))
        except OSError as e:
            # be happy if someone already created the path
            import errno

            if e.errno != errno.EEXIST:
                raise


        with io.open(
            os.path.join(model_dir, file_name + "_extractor.pkl"), "wb"
        ) as f:
            pickle.dump(self.model, f)


    
    def _load(self, file_name: Text="entity_model", model_dir: Text="entity_model"):
        with io.open(
            os.path.join(model_dir, file_name + "_extractor.pkl"), "rb"
        ) as f:
            self.model = pickle.load(f)
    
    
    
    # predict functions
    def _tokenize_and_tag_text(self,text: Text,entites: Dict=None)->List:
        doc = self.nlp(text)
        return [(w.text,w.tag_) for w in doc]

    def _labels_to_json(self, text,data: List)->Dict:
        entitys = defaultdict(list)
        words = [w.text for w in self.nlp(text)]

        for i in range(len(data)):
            item = data[i]
            word = words[i]
            if len(item.split("B-"))>1 :
                entitys[item.split("B-")[1]].append(word)


        return [{"text":" ".join(entitys[k]),"name":k} for k in entitys]

    
    # train functions
    def _preproccess_training_data(self, data :TrainDataClass)->List:
        new_data = []

        for msg in data.data:
            if msg.entities:
                sent = []
                doc = self.nlp(msg.text)
                ents_words = [w.text for w in self.nlp(" ".join([ent['text'] for ent in msg.entities]).strip())]
                for w in doc:
                    if w.text not in ents_words:
                        sent.append((w.text,w.tag_,'O'))

                for ent in msg.entities:
                    doc2 = self.nlp(ent['text'])
                    idx = 0
                    if len(doc2) > 1:
                        for w in doc2:
                            if idx == 0:
                                idx+=1
                                sent.append((w.text,w.tag_,'B-'+ent['label'].replace(" ","_")))
                            else:
                                sent.append((w.text,w.tag_,'I-'+ent['label'].replace(" ","_")))
                    elif len(doc2)==1:
                        w = list(doc2)[0]
                        sent.append((w.text,w.tag_,'B-'+ent['label'].replace(" ","_")))



                new_data.append(sent)
                
        return new_data
        
    
    #utils
    
    def word2features(self, sent, i):
        word = sent[i][0]
        postag = sent[i][1]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],        
        }
        if i > 0:
            word1 = sent[i-1][0]
            postag1 = sent[i-1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if i < len(sent)-1:
            word1 = sent[i+1][0]
            postag1 = sent[i+1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True

        return features


    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
        return [label for token, postag, label in sent]