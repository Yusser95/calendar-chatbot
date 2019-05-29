import spacy


class NERExtractorClass():
    def __init__(self , model='en'):
        self.nlp = spacy.load(model)
        pass
    
    
    def spacy_parse(self ,text):
        # https://spacy.io/api/annotation#named-entities
        
        doc = self.nlp(text)
        ner = []
        for e in doc.ents:
            tmp = {"text":e.text,"label":e.label_}
            ner.append(tmp)
        return ner
        
    
    def parse(self, text ,method='spacy'):
        if method == 'spacy':
            return self.spacy_parse(text)
        
        return {}