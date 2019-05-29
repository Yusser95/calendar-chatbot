


examples = [

{"text":"Set alarm for 7 AM tomorrow","intent":"set_alarm","entites":[{"text":"7 AM tomorrow","label":"date"}]},
{"text":"Wake me up at 7 AM","intent":"set_alarm","entites":[{"text":"7 AM","label":"date"}]},
# {"text":"Wake me up at 7:00 AM","intent":"set_alarm","entites":[{"text":"7:00 AM","label":"date"}]},
# {"text":"Alarm at 6 tonight","intent":"set_alarm","entites":[{"text":"6 tonight","label":"date"}]},
# {"text":"Alarm at 6 pm tomorrow night","intent":"set_alarm","entites":[{"text":"6 pm tomorrow night","label":"date"}]},
# {"text":"Wake me up in half an hour","intent":"set_alarm","entites":[{"text":"half an hour","label":"date"}]},
# {"text":"Wake me up in 355 minutes","intent":"set_alarm","entites":[{"text":"355 minutes","label":"date"}]},
# {"text":"Set alarm this Tuesday at 8am","intent":"set_alarm","entites":[{"text":"Tuesday at 8am","label":"date"}]},
# {"text":"Set alarm this Tue at 8am","intent":"set_alarm","entites":[{"text":"Tue at 8am","label":"date"}]},
# {"text":"Set alarm on Tue at 8am","intent":"set_alarm","entites":[{"text":"Tue at 8am","label":"date"}]},
# {"text":"Set alarm at 2pm on January 30","intent":"set_alarm","entites":[{"text":"2pm on January 30","label":"date"}]},
# {"text":"Wake me up at quarter to 7 tomorrow morning","intent":"set_alarm","entites":[{"text":"quarter to 7 tomorrow morning","label":"date"}]},
# {"text":"Set alarm for 2 in the afternoon","intent":"set_alarm","entites":[{"text":"2 in the afternoon","label":"date"}]},
# {"text":"Set alarm for 2 in the afternoon tomorrow","intent":"set_alarm","entites":[{"text":"2 in the afternoon tomorrow","label":"date"}]},
# {"text":"Set alarm for 8  at night","intent":"set_alarm","entites":[{"text":"8  at night","label":"date"}]},
{"text":"Wake me up at midnight","intent":"set_alarm","entites":[{"text":"midnight","label":"date"}]},
{"text":"Wake me up at noon","intent":"set_alarm","entites":[{"text":"noon","label":"date"}]},
{"text":"Alarm for 7:30 in the morning","intent":"set_alarm","entites":[{"text":"7:30 in the morning","label":"date"}]},
            
            
{"text":"Remind me about Dad's birthday this Thursday","intent":"set_reminder","entites":[{"text":"this Thursday","label":"date"},{"text":"Dad's birthday","label":"message"}]},
{"text":"Remind me about meeting with Jack coming Monday at 3pm","intent":"set_reminder","entites":[{"text":"coming Monday at 3pm","label":"date"},{"text":"meeting with Jack","label":"message"}]},
{"text":"Set reminder about important meeting on January 23 at 4pm","intent":"set_reminder","entites":[{"text":"January 23 at 4pm","label":"date"},{"text":"important meeting","label":"message"}]},
{"text":"Remind me about Lunch with mum next Thursday at 1pm","intent":"set_reminder","entites":[{"text":"next Thursday at 1pm","label":"date"},{"text":"Lunch with mum","label":"message"}]},
            
            
{"text":"Add to my shopping list a new computer","intent":"shopping_list_add","entites":[{"text":"new computer","label":"item"}]},
{"text":"Remove from my shopping list item number 2","intent":"shopping_list_remove","entites":[{"text":"number 2","label":"item"}]},
{"text":"What is in my shopping list","intent":"shopping_list_show","entites":[]},
{"text":"Show my shopping list","intent":"shopping_list_show","entites":[]},
{"text":"Clear my shopping list","intent":"shopping_list_clear","entites":[]},
            
            
{"text":"Add to my to do list call Jason","intent":"todo_list_add","entites":[{"text":"call Jason","label":"item"}]},
{"text":"Insert into my to do list call Jason","intent":"todo_list_add","entites":[{"text":"call Jason","label":"item"}]},
{"text":"Add to my to do list finish homework","intent":"todo_list_add","entites":[{"text":"finish homework","label":"item"}]},
{"text":"Add to my to do list fix my car","intent":"todo_list_add","entites":[{"text":"fix my car","label":"item"}]},
{"text":"Delete from my to do list item 2","intent":"todo_list_remove","entites":[{"text":"2","label":"item"}]},
{"text":"Remove from my to do list item 2","intent":"todo_list_remove","entites":[{"text":"2","label":"item"}]},
{"text":"Remove from my to do list item number 2","intent":"todo_list_remove","entites":[{"text":"number 2","label":"item"}]},
{"text":"Clear my to do list","intent":"todo_list_clear","entites":[]},
{"text":"What is in my to do list","intent":"todo_list_show","entites":[]},
{"text":"Show my to do list","intent":"todo_list_show","entites":[]},
            
            
{"text":"Call 0352765","intent":"dial","entites":[{"text":"0352765","label":"number"}]},
{"text":"Get me 117 on the line","intent":"dial","entites":[{"text":"117","label":"number"}]},
{"text":"Dial 8827734","intent":"dial","entites":[{"text":"8827734","label":"number"}]},
{"text":"Connect me to 37746","intent":"dial","entites":[{"text":"37746","label":"number"}]}

]







from ml import TrainDataClass
from pprint import pprint


############################################## train data


train_data = TrainDataClass()
train_data._create(examples)

############################################## intents




from ml import IntentFeatureExtractorClass
from ml import EmbeddingIntentExtractorClass

# %%time
# train and save
intent_featurizer = IntentFeatureExtractorClass(train_data)
train_data._extract_intent_text_feature(intent_featurizer)
intent_model = EmbeddingIntentExtractorClass(intent_featurizer)
intent_model._train(train_data)
intent_model._save("intent_model")
pprint("finished training !!")



# %%time
# load and predict
intent_model = EmbeddingIntentExtractorClass(intent_featurizer)
intent_model._load("intent_model")
res = intent_model._predict(examples[0]['text'])
pprint(res)

############################################## entities


from ml import CRFsuiteEntityExtractorClass


# %%time
# train and save
entity_model = CRFsuiteEntityExtractorClass()
entity_model._train(train_data)
entity_model._save("entity_model")
pprint("finished training !!")



# %%time
# load and predict
entity_model = CRFsuiteEntityExtractorClass()
entity_model._load("entity_model")

res = entity_model._predict(examples[0]['text'])
pprint(res)