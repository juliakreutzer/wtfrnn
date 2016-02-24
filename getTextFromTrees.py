import codecs
import sys
#extract the text and the labels from the tree structured file (5 labels)

if __name__ == "__main__":
    dataDir = "cs224d/starter_code/trees"
    trainFile = codecs.open(dataDir+"/train.txt", "r", "utf8")
    testFile = codecs.open(dataDir+"/test.txt", "r", "utf8")
    devFile = codecs.open(dataDir+"/dev.txt", "r", "utf8")

    newDataDir = "data/movies/stanfordSentimentTreebank"
    trainSentFile = codecs.open(newDataDir+"/train.sent.txt", "w", "utf8")
    trainLabelFile = codecs.open(newDataDir+"/train.labels5.txt", "w", "utf8")
    testSentFile = codecs.open(newDataDir+"/test.sent.txt", "w", "utf8")
    testLabelFile = codecs.open(newDataDir+"/test.labels5.txt", "w", "utf8")
    devSentFile = codecs.open(newDataDir+"/dev.sent.txt", "w", "utf8")
    devLabelFile = codecs.open(newDataDir+"/dev.labels5.txt", "w", "utf8")

    inFiles = [trainFile, testFile, devFile]
    sentFiles = [trainSentFile, testSentFile, devSentFile]
    labelFiles = [trainLabelFile, testLabelFile, devLabelFile]

    for f, inFile in enumerate(inFiles):
        for i,line in enumerate(inFile):
            #output format:
            #sent file: sentenceID tokens
            #token file: sentenceID labels

            #input format:
            #(1 (1 (2 Due) (1 (2 to) (1 (1 (1 (2 some) (1...

            if not line.startswith("("):
                print("Not a valid format.")
                sys.exit(-1)

           # print line

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
            #print words
            sentFiles[f].write(str(i)+" "+" ".join(words)+"\n")
            labelFiles[f].write(sentenceLabel+" "+" ".join(wordLabels)+"\n")
            #print sentenceLabel
            #print wordLabels

    trainSentFile.close()
    trainLabelFile.close()
    testSentFile.close()
    testLabelFile.close()
    devSentFile.close()
    devLabelFile.close()
    trainFile.close()
    testFile.close()
    devFile.close()