#read a sentence and transform into full binary tree without phrase-level labels
#input: 0 This is a sentence ., 2 3 4 4 4
#output: ...
# (_ (2 This) (3 is) ) (_ (3 is) (4 a)) (_ (4 a) (4 sentence)) (_ (4 sentence) (4 .))

import sys
import codecs
import numpy as np

OPEN="("
CLOSE=")"
PLACEHOLDERLABEL="9"

def nGramToNodePair(ngram, ngramLabel):
    nodepair = OPEN
    nodepair += PLACEHOLDERLABEL+" "
    nodepair += OPEN
    nodepair += ngramLabel[0]+" "
    nodepair += ngram[0]
    nodepair += CLOSE+" "+OPEN
    nodepair += ngramLabel[1]+" "
    nodepair += ngram[1]
    nodepair += CLOSE
    nodepair += CLOSE
    return nodepair

def combine(el1, el2):
    combinedPre = "(9 "
    combinedSuf = ")"
    return combinedPre + el1+" "+el2+combinedSuf

def combineElements(elements):
    #elements contains elements to combine
    newElements = list()
    for i in xrange(len(elements)-1):
        combined = combine(elements[i], elements[i+1])
        newElements.append(combined)
    return newElements

def buildTree(words, wordLabels, sentLabel):
    numLeaves = len(words)
    numNGrams = numLeaves-1

    nGrams = [(words[i],words[i+1]) for i in xrange(0, numLeaves-1)]
    nGramLabels = [(wordLabels[i],wordLabels[i+1]) for i in xrange(0, numLeaves-1)]
    nodePairs = [nGramToNodePair(nGram, nGramLabel) for nGram, nGramLabel in zip(nGrams, nGramLabels)]

    #recursively combine from nodes till root
    #combining = (0 [0] [1])
    elements = nodePairs
    while len(elements)>1:
        elements = combineElements(elements)
    assert elements[0].count("(") == elements[0].count(")")
    #now insert sentLabel
    elements[0] = elements[0].replace("9",str(sentLabel),1)
    newTree = elements[0]
    return newTree

if __name__ == "__main__":

    sentFile = sys.argv[1]
    labelFile = sys.argv[2]

    treeFile = sentFile+".fulltrees.txt"

    osentFile = codecs.open(sentFile, "r", "utf8")
    olabelFile = codecs.open(labelFile, "r", "utf8")
    otreeFile = codecs.open(treeFile, "w", "utf8")

    for sent, labels in zip(osentFile, olabelFile):
        print "tokens", sent
        print "labels", labels

        labels = labels.split()
        sentLabel = labels[0]
        wordLabels = labels[1:]
        words = sent.split()
        sentId = words[0]
        words = words[1:]
        print sentId, words
        print sentLabel, wordLabels

        assert len(words)==len(wordLabels)

        treeString = buildTree(words, wordLabels, sentLabel)
        otreeFile.write(treeString+"\n")

