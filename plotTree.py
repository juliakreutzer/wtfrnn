import numpy as np
try:
    from igraph import Graph, plot, configuration
except ImportError:
    print("No plotting of trees, igraph missing")
import codecs

def plotTreeFromString(treeString, colordict, plotFile):
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

if __name__=="__main__":
#plot a tree from brackets format



    colorDict5 = {1: "red", 2: "red", 3: "white", 4: "green", 5: "green", 0:"grey"}

    inFile="/home/julia/Dokumente/Uni/WS2015/semparse/wtfrnn-repo/trees/train.full.txt.5"
    opened=codecs.open(inFile,"r","utf8")

    l = 5
    c = 0
    for line in opened: #one tree per line
        if len(line)<=1:
            continue

        g = Graph()
        c+=1
        plotFile = "plots/trees/testsent.full.%d.png" % c
        if c>l:
            break
        plotTreeFromString(line, colorDict5, plotFile)
