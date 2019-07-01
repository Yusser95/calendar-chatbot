from typing import Any, Dict, List, Optional, Text, Tuple
import spacy

from .FeatureExtractorClass import FeatureExtractorClass
from .TrainDataClass import TrainDataClass

class IntentFeatureExtractorClass(FeatureExtractorClass):
    def __init__(self ,data: TrainDataClass, method: Text='CountVectorizer')->None:
        data = [d.text for d in data.data]
        
        if method == 'spacy':
            self.nlp = spacy.load('en')
            
        self.method = method
        self._train(data)
        
    
    def _train(self, data: List[Text])->None:
        if self.method == 'spacy':
            pass
        else:
            from sklearn.feature_extraction.text import CountVectorizer
        
            corpus = [e for e in data]
            self.model = CountVectorizer()
            self.model.fit(corpus)
            
    def _save(self, dir_path: Text="intent_model")->Text:
        if self.method == 'spacy':
            pass
        else:
            with io.open(
                os.path.join(dir_path, "text_featre_model.pkl"), "wb"
            ) as f:
                pickle.dump(self.model, f)
                
        return "saved model to {} !".format(os.path.join(dir_path, "text_featre_model.pkl"))
            
    def _get_text_features(self, text: Text)->List:
        if self.method == 'spacy':
            return self.nlp(text).vector
        else:
            if self.model:
                return self.model.transform([text]).toarray()[0]