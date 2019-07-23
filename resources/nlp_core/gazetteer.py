from nltk.tokenize import word_tokenize
import json
import spacy


nlp = spacy.load("en_core_web_sm")
gazetteer = {}
# gazetteer = {
# 	"PERSON":{"words":["client","gab","developer","programmer","general manager","Employee","human resource","chief executive officer" ,"manager"],"symbols":["HR","GM","CEO"]}
# }


def load_gazetteer():
	global gazetteer

	with open("resources/nlp_core/gazetteer.json" , "rb") as f:
		gazetteer = json.load(f)		


def save_gazetteer():
	with open("resources/nlp_core/gazetteer.json" , "w") as f:
		json.dump(gazetteer, f)

def gazetteer_add(word , ent , ent_type='words'):

	if type(word) == str:
		word = word[0].capitalize() + word[1:].lower()
		gazetteer[ent][ent_type].append(word)
	elif type(word) == list:
		word = [w[0].capitalize() + w[1:].lower() for w in word]
		gazetteer[ent][ent_type].extend(word)

	save_gazetteer()


def gazetteer_tag(text, ner):
	doc = nlp(text)
	words = [w.text for w in doc]
	print(ner)

	tagged_text = []
	for k in gazetteer:
		for w in gazetteer[k]["words"]:
			for word in doc:
				if all([True if w.lower() not in t['text'].lower() or t['label'] == 'O' else False for t in ner ]):
					# print(word.text,word.pos_)
					if w.lower() in word.text.lower():
						temp = {"text":word.text ,"label":k}
						tagged_text.append(temp)

		for w in gazetteer[k]["symbols"]:
			if all([True if w.lower() not in t['text'].lower() or t['label'] == 'O' else False for t in ner ]):
				if w in words:
					temp = {"text":w ,"label":k}
					tagged_text.append(temp)

		for w in gazetteer[k]["names"]:
			if all([True if w.lower() not in t['text'].lower() or t['label'] == 'O' else False for t in ner ]):
				if w in words:
					temp = {"text":w ,"label":k}
					tagged_text.append(temp)

	return tagged_text






load_gazetteer()