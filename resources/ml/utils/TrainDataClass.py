from typing import Any, Dict, List, Optional, Text, Tuple
from .FeatureExtractorClass import FeatureExtractorClass
from .MessageClass import MessageClass

     
class TrainDataClass():
    def __init__(self,data: List[MessageClass] = None)->None:
        self.data = data
    
    def _create(self, request: List[Dict])->None:
        self.data = []
        for d in request:
            msg = MessageClass(d['text'],d['intent'],d['entites'])
            self.data.append(msg)
    
    def _extract_intent_text_feature(self, featurizer: FeatureExtractorClass)->None:
        for msg in self.data:
            msg._get_text_features(featurizer)
        pass