import spacy


class RelationExtractorClass():
    def __init__(self , model='en'):
        self.nlp = spacy.load(model)
        pass
    
    
    def spacy_parse(self ,text):        
        relations = []
        doc = self.nlp(text)
        for w in doc:
            tmp = (w.text,w.dep_,w.head.text)
            relations.append(tmp)

        return relations
        
    
    def parse(self, text ,method='spacy'):
        if method == 'spacy':
            return self.spacy_parse(text)
        
        return {}