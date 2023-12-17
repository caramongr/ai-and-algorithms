from glob import glob
import textract
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


word_files = glob('data/gfsr_docs/docx/*.docx')
text = textract.process(word_files[0])
text = text.decode('utf-8')
translator = str.maketrans('', '', string.punctuation + string.digits)
text = text.translate(translator)
#print(text[:200])

en_stopwords = stopwords.words('english')
en_stopwords = set(en_stopwords)

print(en_stopwords)

words = text.lower().split()
words = [w for w in words if w not in en_stopwords and len(w) > 3]