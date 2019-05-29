from typing import Any, Dict, List, Optional, Text, Tuple
from .FeatureExtractorClass import FeatureExtractorClass


class MessageClass():
    def __init__(self,text: Text, intent: Text, entities: List[Dict])->None:
        self.text = text
        self.intent = intent
        self.entities = entities
        pass
    
    def _set_text_features(self, features: List)->None:
        self.intent_text_features = feartures
        pass
    
    def _get_text_features(self, featurizer: FeatureExtractorClass)->None:
        self.intent_text_features = featurizer._get_text_features(self.text)
        pass  