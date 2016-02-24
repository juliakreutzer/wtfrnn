import collections
UNK = 'UNK'
# This file contains the dataset in a useful way. We populate a list of Trees to train/test our Neural Nets such that each Tree contains any number of Node objects.

# The best way to get a feel for how these objects are used in the program is to drop pdb.set_trace() in a few places throughout the codebase
# to see how the trees are used.. look where loadtrees() is called etc..
try:
    from igraph import Graph, plot
except ImportError:
    print("No plotting of trees, igraph missing")

class Node: # a node in the tree
    def __init__(self,label,word=None):
        self.label = label
        self.prediction = None
        self.word = word # NOT a word vector, but index into L.. i.e. wvec = L[:,node.word]
        self.parent = None # reference to parent
        self.left = None # reference to left child
        self.right = None # reference to right child
        self.isLeaf = False # true if I am a leaf (could have probably derived this from if I have a word)
        self.isRoot = False # true if root of the tree
        self.fprop = False # true if we have finished performing fowardprop on this node (note, there are many ways to implement the recursion.. some might not require this flag)
        self.hActs1 = None # h1 from the handout
        self.hActs2 = None # h2 from the handout (only used for RNN2)
        self.probs = None # yhat
        self.prediction = None #prediction of model

class Tree:

    def __init__(self,treeString,openChar='(',closeChar=')'):
        tokens = []
        self.open = '('
        self.close = ')'
        for toks in treeString.strip().split():
            tokens += list(toks)
        self.root = self.parse(tokens)
        self.root.isRoot = True

    def parse(self, tokens, parent=None):
        assert tokens[0] == self.open, "Malformed tree"
        assert tokens[-1] == self.close, "Malformed tree"

        split = 2 # position after open and label
        countOpen = countClose = 0

        if tokens[split] == self.open: 
            countOpen += 1
            split += 1
        # Find where left child and right child split
        while countOpen != countClose:
            if tokens[split] == self.open:
                countOpen += 1
            if tokens[split] == self.close:
                countClose += 1
            split += 1

        # New node
        node = Node(int(tokens[1])) # zero index labels

        node.parent = parent 

        # leaf Node
        if countOpen == 0:
            node.word = ''.join(tokens[2:-1]).lower() # lower case?
            node.isLeaf = True
            return node

        node.left = self.parse(tokens[2:split],parent=node)
        node.right = self.parse(tokens[split:-1],parent=node)

        return node


def leftTraverseWithReturn(root,nodeFn=None,args=None):
    outputs = list()
    outputs.append(nodeFn(root,args))
    if root.left is not None:
        outputs.append(leftTraverseWithReturn(root.left,nodeFn,args))
    if root.right is not None:
        outputs.append(leftTraverseWithReturn(root.right,nodeFn,args))
    return outputs


def leftTraverseSum(root,nodeFn=None,args=None):
    outputs = 0
    outputs += nodeFn(root,args)
    if root.left is not None:
        outputs += leftTraverseSum(root.left,nodeFn,args)
    if root.right is not None:
        outputs += leftTraverseSum(root.right,nodeFn,args)
    return outputs

def leftTraverse(root,nodeFn=None,args=None):
    """
    Recursive function traverses tree
    from left to right. 
    Calls nodeFn at each node
    """
    nodeFn(root,args)
    if root.left is not None:
        leftTraverse(root.left,nodeFn,args)
    if root.right is not None:
        leftTraverse(root.right,nodeFn,args)


def countWords(node,words):
    if node.isLeaf:
        words[node.word] += 1

def clearFprop(node,words):
    node.fprop = False

def mapWords(node,wordMap):
    if node.isLeaf:
        if node.word not in wordMap:
            node.word = wordMap[UNK]
        else:
            node.word = wordMap[node.word]
    
def getLabelList(node,args):
    return node.label

def getLabelsAndWords(node,id2word):
    string = str(node.label)
    if node.isLeaf:
        word = id2word.get(node.word,"UNK")
        string+=" "+word
    return string

def getPredictionsAndWords(node,id2word):
    string = str(node.prediction)
    if node.isLeaf:
        word = id2word.get(node.word,"UNK")
        string+=" "+word
    return string

def countLeaves(node,args):
    if node.isLeaf:
        return 1
    else:
        return 0

def loadWordMap():
    import cPickle as pickle
    
    with open('wordMap.bin','r') as fid:
        return pickle.load(fid)

def buildWordMap():
    """
    Builds map of all words in training set
    to integer values.
    """

    import cPickle as pickle
    file = 'trees/train.txt'
    print "Reading trees to build word map.."
    with open(file,'r') as fid:
        trees = [Tree(l) for l in fid.readlines()]

    print "Counting words to give each word an index.."
    
    words = collections.defaultdict(int)
    for tree in trees:
        leftTraverse(tree.root,nodeFn=countWords,args=words)
    
    wordMap = dict(zip(words.iterkeys(),xrange(len(words))))
    wordMap[UNK] = len(words) # Add unknown as word
    
    print "Saving wordMap to wordMap.bin"
    with open('wordMap.bin','w') as fid:
        pickle.dump(wordMap,fid)

def loadTrees(dataSet='train', wordMap=None):
    """
    Loads training trees. Maps leaf node words to word ids.
    """
    if wordMap is None:
        wordMap = loadWordMap()
    file = 'trees/%s.txt'%dataSet
    print "Loading %sing trees.."%dataSet
    with open(file,'r') as fid:
        trees = [Tree(l) for l in fid.readlines()]
    for tree in trees:
        leftTraverse(tree.root,nodeFn=mapWords,args=wordMap)
    return trees

def tree2string(tree, wordMap, prediction=False):
    """
    Get a string representation of the tree
    :param tree:
    :param wordMap:
    :param prediction: if True node labels are predictions of current model, not true labels
    :return:
    """
    id2word = {v:k for k,v in wordMap.iteritems()}
    if prediction==False:
        labelArray=leftTraverseWithReturn(tree.root,nodeFn=getLabelsAndWords,args=id2word)
    else:
        labelArray=leftTraverseWithReturn(tree.root,nodeFn=getPredictionsAndWords,args=id2word)
    string = labelArray.__str__()
    string = str.replace(string,"[", "(")
    string = str.replace(string,"]", ")")
    string = str.replace(string,",", "")
    string = str.replace(string,"'", "") #FIXME only replace if two '' in node with regex
    string = str.replace(string,'"', "")
    return string

def treeString2plot(treeString, colordict, plotFile):
    """
    Plots a tree from the 'brackets tree' format
    :param treeString:
    :param colordict: defines the colors of the nodes by label (e.g. 1 to 5)
    :param plotFile: output file (.png)
    :return:
    """
    g = Graph()
    splitted = treeString.split("(")

    level = -1
    parents = dict()
    parentIds = dict()
    levelCount = dict()
    for part in splitted:
        if len(part)<1:
            continue
        else: #label follows
            level+=1
            count = levelCount.get(level,0)
            levelCount[level] = count+1
            #print "level %d" % level
            label = part[0]
            #print part.split()
            if len(part.split())>1: #leaf node
                label, wordPlusEnding = part.split()
                #print part, "at leaf"
                endings = wordPlusEnding.count(")")
                word = wordPlusEnding.strip(")")
                g.add_vertex(label=word, color=colordict[int(label)])
                #print "added node %d" % (len(g.vs)-1)
                currentNode = len(g.vs)-1
                p = parents[level-1]
                g.add_edge(currentNode,p)#add edge to parent
                #print "added edge %d-%d" % (len(g.vs)-1, parentIds[level-1])
                level-=endings
                #print "word", word
            else:
                g.add_vertex(label=label, color=colordict[int(label)])
                currentNode = g.vs[len(g.vs)-1]
                #print "added node %d" % (len(g.vs)-1)
                if level != 0:
                    p = parents[level-1]
                    g.add_edge(currentNode,p)#add edge to parent
                    #print "added edge %d-%d" % (len(g.vs)-1, parentIds[level-1])

                parent = currentNode
                parentId = len(g.vs)-1
                parents[level] = parent
                parentIds[level] = parentId
                print parentIds

        print g.summary()
        layout = g.layout_reingold_tilford(mode="in", root=0)
        plot(g, plotFile, layout=layout, bbox = (2000, 1000), margin = 100)


if __name__=='__main__':
    buildWordMap()
    """
    wordMap = loadWordMap()
    #print wordMap
    id2word = {v:k for k,v in wordMap.iteritems()}
    train = loadTrees()

    #from tree to original data format
    labelArrays = list()
    for tree in train:
        #leftTraverse(tree.root, nodeFn=mapWords, args=wordMap)
        labelArray=leftTraverseWithReturn(tree.root,nodeFn=getLabelsAndWords,args=id2word)
        labelArrays.append(labelArray)

    treeStrings = list()
    for labelArray in labelArrays:
        string = labelArray.__str__()
        #print string
        string = str.replace(string,"[", "(")
        string = str.replace(string,"]", ")")
        string = str.replace(string,",", "")
        string = str.replace(string,"'", "") #FIXME only replace if two '' in node with regex
        string = str.replace(string,'"', "")
        treeStrings.append(string)

    print "\n".join(treeStrings[:3])
    """
