import spacy

from .ChunkExtractorClass import ChunkExtractorClass
from .NERExtractorClass import NERExtractorClass
from .TagExtractorClass import TagExtractorClass
from .RelationExtractorClass import RelationExtractorClass


class NLPClass():
    def __init__(self, model = 'en_core_web_sm'):
        self.nlp = spacy.load(model)
        self.chunker = ChunkExtractorClass(self.nlp)
        self.ner_extractor = NERExtractorClass(self.nlp)
        self.tags_extractor = TagExtractorClass(self.nlp)
        self.rels_extractor = RelationExtractorClass(self.nlp)
    
    
    def spacy_parse(self ,text):        
        tmp = {
            "ner":self.chunker.spacy_parse(),
            "chunks":self.ner_extractor.spacy_parse(),
            "tags":self.tags_extractor.spacy_parse(),
            "relations":self.rels_extractor.spacy_parse()
        }
        return tmp
    
    def default_parse(self ,text):        
        tmp = {
            "ner":self.ner_extractor.parse(text , method="cust_PERSON_spacy"),
            "duckling":self.ner_extractor.duckling_parse(text),
            "chunks":self.chunker.parse(text),
            "tags":self.tags_extractor.parse(text),
            "relations":self.rels_extractor.parse(text)
        }
        return tmp
        
    
    def parse(self, text ,method='default'):
        if method == 'spacy':
            return self.spacy_parse(text)
        elif method == 'default':
            return self.default_parse(text)
        
        return {}