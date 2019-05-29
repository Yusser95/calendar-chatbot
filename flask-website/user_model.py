

import json 


from resources.ml import TrainDataClass
from pprint import pprint
from resources.ml import IntentFeatureExtractorClass
from resources.ml import EmbeddingIntentExtractorClass
from resources.ml import CRFsuiteEntityExtractorClass
from resources.nlp_core import NLPClass
from pprint import pprint
import time



class user_model_class():
	def __init__(self , data_path = "examples.json"):
		self.nlp = NLPClass()

		with open(data_path, "r") as f:
			examples = json.load(f)

			self.train_data = TrainDataClass()
			self.train_data._create(examples)

			print("finished reading train_data : ( {} ) example  !! ".format(str(len(self.train_data.data))))



		self.intent_featurizer = IntentFeatureExtractorClass(self.train_data)
		self.intent_model = EmbeddingIntentExtractorClass(self.intent_featurizer)
		self.entity_model = CRFsuiteEntityExtractorClass()

	def _train_models(self):
		tic = time.time()
		self.train_data._extract_intent_text_feature(self.intent_featurizer)
		self.intent_model._train(self.train_data)
		self.entity_model._train(self.train_data)
		toc = time.time()

		print("finished training in ( {} ) seconds !! ".format(str(toc-tic)))


	def _save_models(self):
		tic = time.time()
		self.intent_model._save("intent_model")
		self.entity_model._save("entity_model")
		toc = time.time()

		print("finished saving in ( {} ) seconds !! ".format(str(toc-tic)))

	def _load_models(self):
		tic = time.time()
		self.intent_model._load("intent_model")
		self.entity_model._load("entity_model")
		toc = time.time()

		print("finished loading in ( {} ) seconds !! ".format(str(toc-tic)))


	def _parse_text(self, text):
		tic = time.time()

		data = {}
		data['intent'] = self.intent_model._predict(text)
		data['custum_ents'] = self.entity_model._predict(text)

		tmp = self.nlp.parse(text)
		for k in tmp:
			data[k] = tmp[k]


		toc = time.time()

		print("finished parsing in ( {} ) seconds !! ".format(str(toc-tic)))

		return data








# model = user_model_class()
# model._train_models()
# model._save_models()
# model._load_models()
# res = model._parse_text("set alarm at 7 am")
# pprint(res)



############## stats

# finished reading train_data : ( 28 ) example  !! 
# finished training in ( 7.717316150665283 ) seconds !! 
# finished saving in ( 0.14092111587524414 ) seconds !! 
# finished loading in ( 0.27147579193115234 ) seconds !! 
# finished parsing in ( 0.14717411994934082 ) seconds !! 



############## result

# {'chunks': [alarm],
#  'custum_ents': ['O', 'O', 'O', 'B-date', 'I-date'],
#  'duckling': [{'dim': 'time',
#                'end': 17,
#                'start': 10,
#                'text': 'at 7 am',
#                'value': {'grain': 'hour',
#                          'others': [{'grain': 'hour',
#                                      'value': datetime.datetime(2019, 5, 30, 7, 0, tzinfo=tzoffset(None, 10800))},
#                                     {'grain': 'hour',
#                                      'value': datetime.datetime(2019, 5, 31, 7, 0, tzinfo=tzoffset(None, 10800))},
#                                     {'grain': 'hour',
#                                      'value': datetime.datetime(2019, 6, 1, 7, 0, tzinfo=tzoffset(None, 10800))}],
#                          'value': datetime.datetime(2019, 5, 30, 7, 0, tzinfo=tzoffset(None, 10800))}}],
#  'intent': {'intent': {'confidence': 0.89256751537323, 'name': 'set_alarm'},
#             'intent_ranking': [{'confidence': 0.89256751537323,
#                                 'name': 'set_alarm'},
#                                {'confidence': 0.20708110928535461,
#                                 'name': 'shopping_list_clear'},
#                                {'confidence': 0.11903510987758636,
#                                 'name': 'dial'},
#                                {'confidence': 0.06966584175825119,
#                                 'name': 'todo_list_remove'},
#                                {'confidence': 0.03715144097805023,
#                                 'name': 'todo_list_clear'},
#                                {'confidence': 0.016141414642333984,
#                                 'name': 'shopping_list_add'},
#                                {'confidence': 0.0,
#                                 'name': 'shopping_list_show'},
#                                {'confidence': 0.0,
#                                 'name': 'shopping_list_remove'},
#                                {'confidence': 0.0, 'name': 'todo_list_show'},
#                                {'confidence': 0.0, 'name': 'todo_list_add'}]},
#  'ner': [{'label': 'TIME', 'text': '7 am'}],
#  'relations': [('set', 'csubj', 'am'),
#                ('alarm', 'dobj', 'set'),
#                ('at', 'prep', 'set'),
#                ('7', 'pobj', 'at'),
#                ('am', 'ROOT', 'am')],
#  'tags': [('set', 'VBN'),
#           ('alarm', 'NN'),
#           ('at', 'IN'),
#           ('7', 'CD'),
#           ('am', 'NN')]}






