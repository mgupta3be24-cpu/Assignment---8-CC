
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')


Topic = """I recently started learning guitar.
Not only does it align with my interests, but I also enjoy playing it.
Whenever I feel frustrated, I pick up my guitar and start playing it.
The feeling is relaxing, and I never get bored of it.
I learned more than 10 songs in just 3 months, which requires effort and consistency."""

#i
words = re.findall(r'\b[a-zA-Z]+\b', Topic.lower())
print("Words containing only alphabets:")
print(words)

#ii
stop_words = set(stopwords.words("english"))
filtered_words = [word for word in words if word not in stop_words]

print("\nAfter Stopword Removal:")
print(filtered_words)

#iii
ps = PorterStemmer()
stemmed_words = [ps.stem(word) for word in filtered_words]

print("\nStemmed Words:")
print(stemmed_words)

#iv
lemmatizer = WordNetLemmatizer()
lemmatized_words = [
    lemmatizer.lemmatize(word) for word in filtered_words
]

print("\nLemmatized Words:")
print(lemmatized_words)

#v
print("\nComparison:")
for word, stem, lemma in zip(filtered_words, stemmed_words, lemmatized_words):
    print(word, "-> Stem:", stem, "| Lemma:", lemma)

print("\nExplanation:")
print("Stemming reduces words by removing suffixes.")
print("Lemmatization uses a lexical database to find the base form of words.")
