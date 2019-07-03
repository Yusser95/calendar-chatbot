import spacy
from duckling import DucklingWrapper, Dim
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from .gazetteer import gazetteer_tag

class NERExtractorClass():
    def __init__(self , model):
        self.nlp = model
        self.duckling_wrapper = DucklingWrapper(parse_datetime=True)
        self.stanford_ner = StanfordNERTagger('/Users/mac/stanford-tools/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz',
                       '/Users/mac/stanford-tools/stanford-ner-2018-10-16/stanford-ner.jar',
                       encoding='utf-8')



    def spacy_extract_persons_from_noun_chunks_and_NNP(self, doc):
        persons1 = []
        nsubjs = []
        for w in doc:
            if "nsubj" == w.dep_:
                nsubjs.append(w.text)
        for w in doc.noun_chunks:
            for sub in nsubjs:
                if sub in w.text:
                    persons1.append({"text":w.text,"label":"PERSON"})
                    
        persons2 = [{"text":str(x),"label":"PERSON"} for x in doc if x.tag_ == "NNP"]
        

        persons = persons1+persons2
        persons = [dict(t) for t in {tuple(d.items()) for d in persons}]
        
                    
        return persons
    
    def cust_person_spacy_parse(self ,text):
        # https://spacy.io/api/annotation#named-entities
        
        doc = self.nlp(text)
        ner = []
        for e in doc.ents:
            if e.label_ != "PERSON":
                tmp = {"text":e.text,"label":e.label_}
                ner.append(tmp)

        p_ner = self.spacy_extract_persons_from_noun_chunks_and_NNP(doc)

        ner += p_ner
        return ner



    def spacy_extract_nouns(self, text):
        doc= self.nlp(text)
        persons = []
        for w in doc:
            if w.pos_ == "NOUN":
                tmp = {"text":w.text,"label":"PERSON"}
                persons.append(tmp)

        return persons



    def gazetteer_parse(self, text):
        ner = gazetteer_tag(text)
        return ner

    def cust_person_stanford_parse(self ,text):
        # https://spacy.io/api/annotation#named-entities
        
        ner = self.stanford_parse(text)
        p_ner = self.spacy_extract_nouns(text)
        g_ner = self.gazetteer_parse(text)

        ner += p_ner
        ner += g_ner
        return ner


    def spacy_parse(self ,text):
        # https://spacy.io/api/annotation#named-entities
        
        doc = self.nlp(text)
        ner = []
        for e in doc.ents:
            tmp = {"text":e.text,"label":e.label_}
            ner.append(tmp)
        return ner

    def stanford_parse(self ,text):
        tokenized_text = word_tokenize(text)
        classified_text = self.stanford_ner.tag(tokenized_text)
        ner = []
        for w,t in classified_text:
            tmp = {"text":w,"label":t}
            ner.append(tmp)
        return ner

    def duckling_parse(self, text):
        weekend = 'by the end of the weekend'
        asap = 'the end of the day'

        text = text.lower()

        text += " "

        text = text.replace("the end of the week ",weekend).replace("the end of week ",weekend).replace("end of week ",weekend).replace("end of the week ",weekend)
        text = text.replace("asap",asap).replace("as soon as possible",asap)

        result = self.duckling_wrapper.parse_time(text)
        return result

        
    
    def parse(self, text ,method='spacy'):
        if method == 'spacy':
            return self.spacy_parse(text)
        elif method == 'stanford':
            return self.stanford_parse(text)
        elif method == 'gazetteer':
            return self.gazetteer_parse(text)
        elif method == 'cust_PERSON_spacy':
            return self.cust_person_spacy_parse(text)
        elif method == 'cust_PERSON_stanford':
            return self.cust_person_stanford_parse(text)
        elif method == 'dickling':
            return self.duckling_parse(text)
        
        return {}