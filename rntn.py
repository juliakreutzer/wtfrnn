import collections

import numpy as np

np.seterr(over='ignore',under='ignore')

class RNTN:

    def __init__(self,wvecDim,outputDim,numWords,mbSize=30,rho=1e-6,acti="tanh",init=0.01,partial=False):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho
        self.acti = acti
        self.init = init
        self.partial = partial

        print "RNTN using only partial phrase-level feedback?", self.partial

    def initParams(self, embedding=None):
        np.random.seed(12341)
        
        # Word vectors
        if embedding is None:
            self.L = self.init*np.random.randn(self.wvecDim,self.numWords)
        else:
            self.L = embedding #pre-trained, matrix
        # Hidden activation weights
        self.V = self.init*np.random.randn(self.wvecDim,2*self.wvecDim,2*self.wvecDim)
        self.W = self.init*np.random.randn(self.wvecDim,self.wvecDim*2)
        self.b = np.zeros((self.wvecDim))

        # Softmax weights
        self.Ws = self.init*np.random.randn(self.outputDim,self.wvecDim)
        self.bs = np.zeros((self.outputDim))

        self.stack = [self.L, self.V, self.W, self.b, self.Ws, self.bs]

        # Gradients
        self.dV = np.empty((self.wvecDim,2*self.wvecDim,2*self.wvecDim))
        self.dW = np.empty(self.W.shape)
        self.db = np.empty((self.wvecDim))
        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))

    def costAndGrad(self,mbdata,test=False):
        """
        Each datum in the minibatch is a tree.
        Forward prop each tree.
        Backprop each tree.
        Returns
           cost
           Gradient w.r.t. W, Ws, b, bs, V
           Gradient w.r.t. L in sparse form.

        or if in test mode
        Returns
           cost, correctArray, guessArray, total

        """
        cost = 0.0
        correct = []
        guess = []
        total = 0.0

        self.L,self.V,self.W,self.b,self.Ws,self.bs = self.stack

        # Zero gradients
        self.dW[:] = 0
        self.db[:] = 0
        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dV[:] = 0
        self.dL = collections.defaultdict(self.defaultVec)

        # Forward prop each tree in minibatch
        for tree in mbdata:
            c,tot = self.forwardProp(tree.root,correct,guess)
            cost += c
            total += tot
        if test:
            return (1./len(mbdata))*cost,correct,guess,total

        # Back prop each tree in minibatch
        for tree in mbdata:
            self.backProp(tree.root)

        # scale cost and grad by mb size
        scale = (1./self.mbSize)
        for v in self.dL.itervalues():
            v *=scale

        # Add L2 Regularization for W, Ws and V
        cost += (self.rho/2)*np.sum(self.W**2)
        cost += (self.rho/2)*np.sum(self.Ws**2)
        cost += (self.rho/2)*np.sum(self.V**2)

        return scale*cost,[self.dL,scale*(self.dV + self.rho*self.V), scale*(self.dW + self.rho*self.W),scale*self.db,
                           scale*(self.dWs+self.rho*self.Ws),scale*self.dbs]

    def forwardProp(self,node,correct=[],guess=[]):
        epsilon = 1e-10
        cost = total = 0.0

        # Recursion
        if not node.isLeaf:
            left_cost, left_total = self.forwardProp(node.left, correct, guess)
            right_cost, right_total = self.forwardProp(node.right, correct, guess)
            cost += (left_cost + right_cost)
            total += (left_total + right_total)
            # Compute hidden layer
            node.hActs1 = np.dot(self.W, np.hstack([node.left.hActs1, node.right.hActs1])) + self.b #composition of child nodes
            node.hActs1 += np.dot(np.dot(np.hstack([node.left.hActs1, node.right.hActs1]),self.V),
                                  np.hstack([node.left.hActs1, node.right.hActs1])) #tensor composition
            if self.acti == "relu":
                node.hActs1[node.hActs1 < 0] = 0 #ReLu
            elif self.acti == "tanh":
                node.hActs1 = np.tanh(node.hActs1) #tanh
            else:
                print "no valid activation function given (tanh, relu), taking tanh as default"
                node.hActs1 = np.tanh(node.hActs1) #tanh
        else:
            node.hActs1 = self.L[:,node.word]


        if (node.isLeaf or node.isRoot or not self.partial) and node.label!=9:

            # Softmax (taken from lecture slides)
            node.probs = np.dot(self.Ws, node.hActs1) + self.bs
            node.probs -= np.max(node.probs) #prevent overflows, underflows are ignored since their contribution is negligible
            #http://rodresearch.blogspot.de/2011/08/avoiding-overflow-problem-in-softmax.html
            node.probs = np.exp(node.probs)
            node.probs = node.probs/np.sum(node.probs)

            # Calculate cross entropy cost

            # if node.isLeaf:
            #     print "leaf"
            # if node.isRoot:
            #     print "root"
            # if not self.partial:
            #     print "full"

            correct.append(node.label)
            #print "correct", node.label
            node.prediction = np.argmax(node.probs)
            guess.append(node.prediction)
            #print "predicted", node.label

            try:
                #print node.probs[node.label], node.label
                if node.probs[node.label]==0: #problematic for log
                    node.probs[node.label]+=epsilon #doesn't change prediction
                cost += -np.log(node.probs[node.label])
            except IndexError:
                print "something went wrong", "label", node.label, "probs", node.probs

            # We performed forward propagation
            node.fprop = True

        return cost, total + 1


    def backProp(self,node,error=None):
        if (node.isLeaf or node.isRoot or not self.partial) and node.label!=9:
            # Softmax grad
            deltas = node.probs #calculated in FF
            deltas[node.label] -= 1.0 #y_hat - y #y is 0 for all but node.label indices #delta_softmax (delta_3)
            self.dWs += np.outer(deltas, node.hActs1)
            self.dbs += deltas

            # Add deltas from above
            deltas = np.dot(self.Ws.T, deltas)
        else:
            deltas = np.zeros(shape=error.shape)

        #incoming error
        if error is not None:
            deltas += error

        if self.acti == "relu":
            deltas *= (node.hActs1 != 0) #deriv of relu  (delta_2)
        elif self.acti == "tanh":
            deltas *= 1 - np.tanh(node.hActs1)**2  #deriv of tanh  tanh'(x)= 1 - tan2(x)
        else:
            print "no valid activation function given (tanh, relu), taking tanh as default"
            deltas *= 1 - np.tanh(node.hActs1)**2  #deriv of tanh  tanh'(x)= 1 - tan2(x)


        # Update word vectors if leaf node
        if node.isLeaf:
            self.dL[node.word] += deltas
            return

        # Update hidden layer weights
        if not node.isLeaf:
            self.dW += np.outer(deltas, np.hstack([node.left.hActs1, node.right.hActs1]))
            self.db += deltas
            # Error signal to children
            deltas = np.dot(self.W.T, deltas) + \
                     np.sum([np.dot(np.dot(deltas[i],(k+k.T)),np.hstack([node.left.hActs1, node.right.hActs1]))
                             for i,k in enumerate(self.V)]) #process tensor slice-wise
            self.backProp(node.left, deltas[:self.wvecDim])
            self.backProp(node.right, deltas[self.wvecDim:])
            self.dV += [np.dot(deltas[i],
                               np.outer(np.hstack([node.left.hActs1, node.right.hActs1]),np.hstack([node.left.hActs1, node.right.hActs1]).T))
                        for i,k in enumerate(self.V)] #update slice-wise

            # Clear nodes
            node.fprop = False
        
    def updateParams(self,scale,update,log=False):
        """
        Updates parameters as
        p := p - scale * update.
        If logs is true, prints root mean square of parameter
        and update.
        """
        if log:
            for P,dP in zip(self.stack[1:],update[1:]):
                pRMS = np.sqrt(np.mean(P**2))
                dpRMS = np.sqrt(np.mean((scale*dP)**2))
                print "weight rms=%f -- update rms=%f"%(pRMS,dpRMS)

        self.stack[1:] = [P+scale*dP for P,dP in zip(self.stack[1:],update[1:])]

        # handle dictionary update sparsely
        dL = update[0]
        for j in dL.iterkeys():
            self.L[:,j] += scale*dL[j]

    def toFile(self,fid):
        import cPickle as pickle
        pickle.dump(self.stack,fid)

    def fromFile(self,fid):
        import cPickle as pickle
        self.stack = pickle.load(fid)

    def check_grad(self,data,epsilon=1e-6):

        cost, grad = self.costAndGrad(data)
        err1 = 0.0
        count = 0.0

        print "Checking dV... (might take a while)"
        for V,dV in zip(self.stack[1:],grad[1:]):
            V = V[...,None,None] # add dimension since bias is flat
            dV = dV[...,None,None]
            for i in xrange(V.shape[0]):
                for j in xrange(V.shape[1]):
                    for k in xrange(V.shape[2]):
                        V[i,j,k] += epsilon
                        costP,_ = self.costAndGrad(data)
                        V[i,j,k] -= epsilon
                        numGrad = (costP - cost)/epsilon
                        err = np.abs(dV[i,j,k] - numGrad)
                        #print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dW[i,j,k],numGrad,err)
                        err1+=err
                        count+=1
        if 0.001 > err1/count:
            print "Grad Check Passed for dV"
        else:
            print "Grad Check Failed for dV: Sum of Error = %.9f" % (err1/count)

        print "Checking dV ... (might take a while)"

        # check dL separately since dict
        dL = grad[0]
        L = self.stack[0]
        err2 = 0.0
        count = 0.0
        print "Checking dL..."
        for j in dL.iterkeys():
            for i in xrange(L.shape[0]):
                L[i,j] += epsilon
                costP,_ = self.costAndGrad(data)
                L[i,j] -= epsilon
                numGrad = (costP - cost)/epsilon
                err = np.abs(dL[j][i] - numGrad)
                #print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dL[j][i],numGrad,err)
                err2+=err
                count+=1

        if 0.001 > err2/count:
            print "Grad Check Passed for dL"
        else:
            print "Grad Check Failed for dL: Sum of Error = %.9f" % (err2/count)

if __name__ == '__main__':

    import tree as treeM
    train = treeM.loadTrees()
    numW = len(treeM.loadWordMap())

    wvecDim = 10
    outputDim = 5

    nn = RNTN(wvecDim,outputDim,numW,mbSize=4)
    nn.initParams()

    mbData = train[:1]
    #cost, grad = nn.costAndGrad(mbData)

    print "Numerical gradient check..."
    nn.check_grad(mbData)






