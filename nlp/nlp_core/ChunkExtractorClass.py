import spacy


class ChunkExtractorClass():
    def __init__(self , model='en'):
        self.nlp = spacy.load(model)
        pass
    
    
    def spacy_parse(self ,text):  
        doc = self.nlp(text)
        noun_chunks = list(doc.noun_chunks)
        return noun_chunks
        
    
    def parse(self, text ,method='spacy'):
        if method == 'spacy':
            return self.spacy_parse(text)
        
        return {}