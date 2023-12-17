from textblob import TextBlob
from textblob import Word

text = 'Today is a beautiful day. Tomorrow looks like bad weather.'

blob = TextBlob(text)

print(blob.sentences)

print(blob.words)

print(blob.tags)

print(blob.noun_phrases)

print(blob.sentiment)

print(blob.sentiment.polarity)

print(blob.sentiment.subjectivity)

print(round(blob.sentiment.polarity, 3))

print(round(blob.sentiment.subjectivity, 3))

happy = Word('happy')

print(happy.definitions)
