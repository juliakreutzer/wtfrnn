import codecs
import sys
from fromTextToFullTree import *

#transform parsed trees into full binary trees


if __name__=="__main__":

    limit=15
    dataDir = "trees"
    trainFile = codecs.open(dataDir+"/train.txt", "r", "utf8")
    testFile = codecs.open(dataDir+"/test.txt", "r", "utf8")
    devFile = codecs.open(dataDir+"/dev.txt", "r", "utf8")

    trainOutFile = codecs.open(dataDir+"/train.full.%d.txt" % limit, "w", "utf8")
    testOutFile = codecs.open(dataDir+"/test.full.%d.txt" % limit, "w", "utf8")
    devOutFile = codecs.open(dataDir+"/dev.full.%d.txt" % limit, "w", "utf8")

    inFiles = [trainFile, testFile, devFile]
    outFiles = [trainOutFile, testOutFile, devOutFile]

    for f, inFile in enumerate(inFiles):
        for i,line in enumerate(inFile):

            print "line", i
            #first filter words from trees

            if not line.startswith("("):
                print("Not a valid format.")
                sys.exit(-1)

            wordLabels = list()
            words = list()
            sentenceLabel = -1
            sentenceLevel = True
            lastLabel = -1
            word = ""

            for j in xrange(len(line)):
                char = line[j]
                #print "char", char
                if char == "(":
                    continue
                elif char == ")":
                    if len(word)>0:
                        words.append(word)
                        #print "new word", word, lastLabel
                        wordLabels.append(lastLabel)
                        word = ""
                elif char.isspace():
                    continue
                else:
                    if char.isdigit(): #label
                        if sentenceLevel:
                            sentenceLabel = char
                            #print "sent label", sentenceLabel
                            sentenceLevel = False
                        else:
                            lastLabel = char #save for later
                    else: #word
                        word += char

            assert len(words) == len(wordLabels)

            if len(words)<limit:

                #now transform to full tree
                #remove phrase-level annotations, keep word-level and sentence-level
                treeString = buildTree(words, wordLabels, sentenceLabel)
                outFiles[f].write(treeString+"\n")
                outFiles[f].flush()
                #print sentenceLabel
                #print wordLabels

    trainFile.close()
    testFile.close()
    devFile.close()
    trainOutFile.close()
    testOutFile.close()
    devOutFile.close()
