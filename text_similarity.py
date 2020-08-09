from string import punctuation
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

language_stopwords = stopwords.words('english')
non_words = list(punctuation)

def remove_stop_words(dirty_text):
	cleaned_text = ''
	for word in dirty_text.split():
		if word in language_stopwords or word in non_words:
			continue
		else:
			cleaned_text += word + ' '
	return cleaned_text

def remove_punctuation(dirty_string):
	for word in non_words:
		dirty_string = dirty_string.replace(word, '')
	return dirty_string

def process_file(file_name):
	file_content = open(file_name, "r").read()
	# All to lower case
	file_content = file_content.lower()
	# Remove punctuation and spanish stopwords
	file_content = remove_punctuation(file_content)
	file_content = remove_stop_words(file_content)
	return file_content

nlp_article = process_file("nlp.txt")
sentiment_analysis_article = process_file("sentiment_analysis.txt")
java_certification_article = process_file("java_cert.txt")

#TF-IDF
vectorizer = TfidfVectorizer ()
X = vectorizer.fit_transform([nlp_article,sentiment_analysis_article,java_certification_article])
#X = count.fit_transform([nlp_article,sentiment_analysis_article,java_certification_article])
similarity_matrix = cosine_similarity(X,X)

print('----------------------------------')
print('Leantechblog article similarity:')
print('----------------------------------')
print(similarity_matrix)

michelle_speech = process_file("michelle_speech.txt")
melania_speech = process_file("melania_speech.txt")

#TF-IDF
vectorizer = TfidfVectorizer ()
X = vectorizer.fit_transform([michelle_speech,melania_speech])
similarity_matrix = cosine_similarity(X,X)

print('-----------------------------------------')
print('Melania and Michelle speeches similarity:')
print('-----------------------------------------')
print(similarity_matrix)