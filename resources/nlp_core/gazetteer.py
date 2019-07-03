from nltk.tokenize import word_tokenize


gazetteer = {
	"PERSON":["client"]
}

def gazetteer_tag(text):
	words = word_tokenize(text)
	tagged_text = []
	for k in gazetteer:
		for w in gazetteer[k]:
			if w.lower() in words:
				temp = {"text":w ,"label":k}
				tagged_text.append(temp)
				
	return tagged_text
