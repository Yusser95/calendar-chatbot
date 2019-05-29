import spacy


class TagExtractorClass():
    def __init__(self , model='en'):
        self.nlp = spacy.load(model)
        pass
    
    
    def spacy_parse(self ,text):        
        tags = []
        doc = self.nlp(text)
        for w in doc:
            tmp = (w.text,w.tag_)
            tags.append(tmp)

        return tags
        
    
    def parse(self, text ,method='spacy'):
        if method == 'spacy':
            return self.spacy_parse(text)
        
        return {}