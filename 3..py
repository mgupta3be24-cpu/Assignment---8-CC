
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

texts = [
    "I recently started learning guitar. I enjoy playing guitar because it aligns with my interests.",
    "Whenever I feel frustrated, I pick up my guitar and start playing. Playing guitar makes me feel relaxed and happy.",
    "I learned more than 10 songs in just 3 months. Learning guitar requires effort, patience and consistency."
]

#i
cv = CountVectorizer()
bow = cv.fit_transform(texts)

bow_df = pd.DataFrame(
    bow.toarray(),
    columns=cv.get_feature_names_out()
)

print("Bag of Words Representation:")
print(bow_df)

print("\n")
#ii
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(texts)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf.get_feature_names_out()
)

print("TF-IDF Scores:")
print(tfidf_df.round(3))


print("\n")
#iii
features = tfidf.get_feature_names_out()

print("Top 3 Keywords from Each Text:")
print("\n")

for i in range(len(texts)):
    scores = tfidf_matrix[i].toarray()[0]
    top_indices = scores.argsort()[-3:][::-1]

    print(f"Text {i+1}:")
    for index in top_indices:
        print(features[index], ":", round(scores[index], 3))
