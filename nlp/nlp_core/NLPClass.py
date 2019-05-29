import spacy

from .ChunkExtractorClass import ChunkExtractorClass
from .NERExtractorClass import NERExtractorClass
from .TagExtractorClass import TagExtractorClass
from .RelationExtractorClass import RelationExtractorClass


class NLPClass():
    def __init__(self):
        self.chunker = ChunkExtractorClass()
        self.ner_extractor = NERExtractorClass()
        self.tags_extractor = TagExtractorClass()
        self.rels_extractor = RelationExtractorClass()
    
    
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
            "ner":self.ner_extractor.parse(text),
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