Topic = (
    "I recently started learning guitar because it matches my interests. "
    "I enjoy playing it whenever I have free time. "
    "Whenever I feel frustrated, I pick up my guitar and start playing, "
    "which helps me relax. "
    "Playing the guitar makes me feel calm, and I've never gotten bored of it. "
    "I learned more than ten songs in just three months. "
    "This achievement required a lot of effort, patience, and consistency."
)

import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

text = re.sub(r'[^\w\s]', '', Topic.lower())
print("\n")
print("Lowercase and Punctuation removed")
print(text)



words = word_tokenize(Topic)
sentences = sent_tokenize(Topic)
print("\n")
print("Word Tokens:")
print(words)

print("\n")
print("Sentence Tokens:")
print(sentences)
print("\n")

print("Using split():")
print(Topic.split())
print("\n")

print("Using word_tokenize():")
print(word_tokenize(Topic))

print("\n")

#4
stop_words = set(stopwords.words("english"))

tokens = word_tokenize(text)
New_tokens = [
    word for word in tokens
    if word.lower() not in stop_words
]
print("After Stopwords Removal:")
print(New_tokens)

print("\n")
freq_dist = FreqDist(New_tokens)
print(freq_dist)

print("\n")

#v

print("\nWord Frequency Distribution:")
for word, frequency in freq_dist.items():
    print(word, ":", frequency)
