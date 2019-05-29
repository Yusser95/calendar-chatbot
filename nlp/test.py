from nlp_core import NLPClass
from pprint import pprint

text = 'find good Cafe near Home San Francisco'
nlp = NLPClass()
tmp = nlp.parse(text)
pprint(tmp)