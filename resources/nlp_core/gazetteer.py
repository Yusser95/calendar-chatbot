from nltk.tokenize import word_tokenize


gazetteer = {
	"PERSON":{"words":["client","gab","developer","programmer","general manager","Employee","human resource","chief executive officer" ,"manager"],"symbols":["HR","GM","CEO"]}
}

def gazetteer_tag(text):
	words = word_tokenize(text)
	tagged_text = []
	for k in gazetteer:
		for w in gazetteer[k]["words"]:
			for word in words:
				if w.lower() in word.lower():
					temp = {"text":word ,"label":k}
					tagged_text.append(temp)
		for w in gazetteer[k]["symbols"]:
			if w in words:
				temp = {"text":w ,"label":k}
				tagged_text.append(temp)
				
	return tagged_text
