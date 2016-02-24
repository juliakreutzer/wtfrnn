from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import sys
import codecs
import re

#remove empty lines
#tokenize and remove stopwords

def preprocess(doc):
    """ Preprocess a document: tokenize words, lowercase, remove stopwords, non-alphabetic characters and empty lines"""
    sw = stopwords.words('english')
    tokenized = [re.sub('[^A-Za-z0-9]+', '', word.lower()) if word.lower() not in sw else "" for word in word_tokenize(doc)] #nltk tokenizer
    specialChars = ["",'', ' ', '\t']
    tokenized[:] = (char for char in tokenized if char not in specialChars and len(char)>2)
    #print tokenized
    return ' '.join(tokenized)

if __name__=="__main__":
    inFile = sys.argv[1]
    opened = codecs.open(inFile, "r", "utf8")
    text = opened.read()
    opened.close()
    preprocessed = preprocess(text)
    outFile = sys.argv[1]+".tok"
    openedOut = codecs.open(outFile, "w", "utf8")
    openedOut.write(preprocessed)
    openedOut.close()
    print "Preprocessed %s and wrote to file %s" % (inFile, outFile)
