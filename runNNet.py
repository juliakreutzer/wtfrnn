import optparse
import cPickle as pickle

import sgd as optimizer
from rntn import RNTN
from rnn import RNN


import tree as tr
import time
import matplotlib.pyplot as plt
import numpy as np
import pdb

from gensim import models
#from sklearn.metrics import confusion_matrix

# This is the main training function of the codebase. You are intended to run this function via command line 
# or by ./run.sh

# You should update run.sh accordingly before you run it!


# TODO:
# Create your plots here

def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--test",action="store_true",dest="test",default=False)

    # Optimizer
    parser.add_option("--minibatch",dest="minibatch",type="int",default=30)
    parser.add_option("--optimizer",dest="optimizer",type="string",
        default="adagrad")
    parser.add_option("--epochs",dest="epochs",type="int",default=50)
    parser.add_option("--step",dest="step",type="float",default=1e-2)
    parser.add_option("--init",dest="init",type="float",default=0.01)

    parser.add_option("--outputDim",dest="outputDim",type="int",default=5)
    parser.add_option("--wvecDim",dest="wvecDim",type="int",default=30)

    parser.add_option("--rho",dest="rho",type="float",default=1e-6)

    parser.add_option("--outFile",dest="outFile",type="string",
        default="models/test.bin")
    parser.add_option("--inFile",dest="inFile",type="string",
        default="models/test.bin")
    parser.add_option("--data",dest="data",type="string",default="train")

    parser.add_option("--model",dest="model",type="string",default="RNN")

    parser.add_option("--maxTrain",dest="maxTrain", type="int", default=-1)
    parser.add_option("--activation",dest="acti", type="string", default="tanh")

    parser.add_option("--partial",action="store_true",dest="partial",default=False)
    parser.add_option("--w2v",dest="w2vmodel", type="string")

    (opts,args)=parser.parse_args(args)


    # make this false if you dont care about your accuracies per epoch, makes things faster!
    evaluate_accuracy_while_training = True

    # Testing
    if opts.test:
        cmfile = opts.inFile + ".confusion_matrix-" + opts.data + ".png"
        test(opts.inFile,opts.data,opts.model,acti=opts.acti)
        return
    
    print "Loading data..."

    embedding = None
    wordMap = None
    if opts.w2vmodel is not None:
        print "Loading pre-trained word2vec model from %s" % opts.w2vmodel
        w2v = models.Word2Vec.load(opts.w2vmodel)
        embedding, wordMap = readW2v(w2v,opts.wvecDim)

    train_accuracies = []
    train_rootAccuracies = []
    dev_accuracies = []
    dev_rootAccuracies = []
    # load training data
    trees = tr.loadTrees('train',wordMap=wordMap)[:opts.maxTrain] #train.full.15
    if opts.maxTrain > -1:
        print "Training only on %d trees" % opts.maxTrain
    opts.numWords = len(tr.loadWordMap())


    if opts.partial==True:
        print "Only partial feedback"

    if (opts.model=='RNTN'):
        nn = RNTN(wvecDim=opts.wvecDim,outputDim=opts.outputDim,numWords=opts.numWords,
                  mbSize=opts.minibatch,rho=opts.rho, acti=opts.acti, init=opts.init, partial=opts.partial)
    elif(opts.model=='RNN'):
        nn = RNN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
    else:
        raise '%s is not a valid neural network so far only RNTN, RNN'%opts.model
    
    nn.initParams(embedding=embedding)

    sgd = optimizer.SGD(nn,alpha=opts.step,minibatch=opts.minibatch,
        optimizer=opts.optimizer)


    dev_trees = tr.loadTrees("dev") #dev.full.15
    for e in range(opts.epochs):
        start = time.time()
        print "Running epoch %d"%e
        sgd.run(trees)
        end = time.time()
        print "Time per epoch : %f"%(end-start)

        with open(opts.outFile,'w') as fid:
            pickle.dump(opts,fid)
            pickle.dump(sgd.costt,fid)
            nn.toFile(fid)
        if evaluate_accuracy_while_training:
            print "testing on training set"
            acc, sacc = test(opts.outFile,"train",opts.model,trees)
            train_accuracies.append(acc)
            train_rootAccuracies.append(sacc)
            print "testing on dev set"
            dacc, dsacc = test(opts.outFile,"dev",opts.model,dev_trees)
            dev_accuracies.append(dacc)
            dev_rootAccuracies.append(dsacc)
            # clear the fprop flags in trees and dev_trees
            for tree in trees:
                tr.leftTraverse(tree.root,nodeFn=tr.clearFprop)
            for tree in dev_trees:
                tr.leftTraverse(tree.root,nodeFn=tr.clearFprop)
            print "fprop in trees cleared"


    if evaluate_accuracy_while_training:
        pdb.set_trace()
        print train_accuracies
        print dev_accuracies

        print "on sentence-level:"
        print train_rootAccuracies
        print dev_rootAccuracies

        # Plot train/dev_accuracies
        plt.figure()
        plt.plot(range(len(train_accuracies)), train_accuracies, label='Train')
        plt.plot(range(len(dev_accuracies)), dev_accuracies, label='Dev')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        # plot.show()
        plt.savefig(opts.outFile + ".accuracy_plot.png")

          # Plot train/dev_accuracies
        plt.figure()
        plt.plot(range(len(train_rootAccuracies)), train_rootAccuracies, label='Train')
        plt.plot(range(len(dev_rootAccuracies)), dev_rootAccuracies, label='Dev')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        # plot.show()
        plt.savefig(opts.outFile + ".sent.accuracy_plot.png")

def readW2v(w2v,wvDim):
    wordMap = dict()
    numWords = len(w2v.index2word)
    if "UNK" not in w2v.index2word:
        numWords += 1
    matrix = np.empty(shape=(wvDim,numWords))
    for i,word in enumerate(w2v.index2word):
        wordMap[word] = i
        matrix[:,i] = w2v[word]
    if "UNK" not in w2v.index2word: #add UNK zeros
        matrix[:,-1] = np.zeros(shape=wvDim)
        wordMap["UNK"] = len(matrix)
    return matrix, wordMap

def test(netFile,dataSet, model='RNN', trees=None, confusion_matrix_file=None):
    if trees==None:
        trees = tr.loadTrees(dataSet)
    assert netFile is not None, "Must give model to test"
    print "Testing netFile %s"%netFile
    with open(netFile,'r') as fid:
        opts = pickle.load(fid)
        _ = pickle.load(fid)
        
        if (model=='RNTN'):
            nn = RNTN(wvecDim=opts.wvecDim,outputDim=opts.outputDim,numWords=opts.numWords,mbSize=opts.minibatch,rho=opts.rho, acti=opts.acti)
        elif(model=='RNN'):
            nn = RNN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
        else:
            raise '%s is not a valid neural network so far only RNTN, RNN'%opts.model
        
        nn.initParams()
        nn.fromFile(fid)

    print "Testing %s..."%model

    cost,correct, guess, total = nn.costAndGrad(trees,test=True)
    correct_sum = 0
    for i in xrange(0,len(correct)):
        correct_sum+=(guess[i]==correct[i])

    correctSent = 0
    for tree in trees:
        sentLabel = tree.root.label
        sentPrediction = tree.root.prediction
        if sentLabel == sentPrediction:
            correctSent += 1


    # Generate confusion matrix
    #if confusion_matrix_file is not None:
    #    cm = confusion_matrix(correct, guess)
    #    makeconf(cm, confusion_matrix_file)

    print "%s: Cost %f, Acc %f, Sentence-Level: Acc %f"%(dataSet,cost,correct_sum/float(total),correctSent/float(len(trees)))
    return (correct_sum/float(total), correctSent/float(len(trees)))


def makeconf(conf_arr, outFile):
    # makes a confusion matrix plot when provided a matrix conf_arr
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')

    width = len(conf_arr)
    height = len(conf_arr[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    indexs = '0123456789'
    plt.xticks(range(width), indexs[:width])
    plt.yticks(range(height), indexs[:height])
    # you can save the figure here with:
    plt.savefig(outFile)
    print "Confusion Matrix written to %s" % outFile
    plt.show()


if __name__=='__main__':
    run()

