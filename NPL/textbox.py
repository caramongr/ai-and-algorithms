from textblob import TextBlob, Word
from textblob.sentiments import NaiveBayesAnalyzer

text = "Today is a beautiful day. Tomorrow looks like bad weather."

blob = TextBlob(text)

print(blob)

print(blob.sentences)

print(blob.words)

print(blob.tags)

print(blob.noun_phrases)

print(blob.sentiment)

for sentence in blob.sentences:
    print(sentence.sentiment)

blob2 = TextBlob(text, analyzer=NaiveBayesAnalyzer())

print(blob2.sentiment)

print("-----------------------------")

for sentence in blob2.sentences:
    print(sentence.sentiment)

# blob.detect_language()

Word = Word("studies").singularize()
print(Word)

sentence = TextBlob('I canot beleive I misspelled that word')

print(sentence.correct())


