import numpy as np
import random
np.seterr(over='ignore',under='ignore')


class SGD:

    def __init__(self,model,alpha=1e-2,minibatch=30,
                 optimizer='sgd'):
        self.model = model
        print "initializing SGD"
        assert self.model is not None, "Must define a function to optimize"
        self.it = 0
        self.alpha = alpha # learning rate
        self.minibatch = minibatch # minibatch
        self.optimizer = optimizer
        if self.optimizer == 'sgd':
            print "Using sgd..."
        elif self.optimizer == 'adagrad':
            print "Using adagrad..."
            epsilon = 1e-8
            self.gradt = [epsilon + np.zeros(W.shape) for W in self.model.stack]
        elif self.optimizer == 'adadelta':
            print "Using adadelta..."
            self.epsilon = 1e-8
            self.adarho = 0.95
            self.accum_grad = [np.zeros(W.shape) for W in self.model.stack]
            self.accum_update = [np.zeros(W.shape) for W in self.model.stack]
        else:
            raise ValueError("Invalid optimizer")

        self.costt = []
        self.expcost = []


    def run(self,trees):
        """
        Runs stochastic gradient descent with model as objective.
        """
        print "running SGD"
        
        m = len(trees)

        # randomly shuffle data
        random.shuffle(trees)

        for i in xrange(0,m-self.minibatch+1,self.minibatch):
            self.it += 1

            mb_data = trees[i:i+self.minibatch]
               
            cost,grad = self.model.costAndGrad(mb_data)

            # compute exponentially weighted cost
            if np.isfinite(cost):
                if self.it > 1:
                    self.expcost.append(.01*cost + .99*self.expcost[-1])
                else:
                    self.expcost.append(cost)

            if self.optimizer == 'sgd':
                update = grad
                scale = -self.alpha

            elif self.optimizer == 'adagrad':
                # trace = trace+grad.^2
                self.gradt[1:] = [gt+g**2 
                        for gt,g in zip(self.gradt[1:],grad[1:])]
                # update = grad.*trace.^(-1/2)
                update =  [g*(1./np.sqrt(gt))
                        for gt,g in zip(self.gradt[1:],grad[1:])]
                # handle dictionary separately
                dL = grad[0]
                dLt = self.gradt[0]
                for j in dL.iterkeys():
                    dLt[:,j] = dLt[:,j] + dL[j]**2
                    dL[j] = dL[j] * (1./np.sqrt(dLt[:,j]))
                update = [dL] + update
                scale = -self.alpha

            elif self.optimizer == 'adadelta':
                # accumulate gradient
                self.accum_grad[1:] = [self.adarho*ag + (1-self.adarho)*g**2
                                       for ag,g in zip(self.accum_grad[1:], grad[1:])]
                # compute update
                update = [- np.sqrt(au + self.epsilon) / np.sqrt(ag + self.epsilon)* g
                          for au,ag,g in zip(self.accum_update[1:],self.accum_grad[1:],grad[1:])]
                # accumulate update
                self.accum_update[1:] = [self.adarho*au + (1-self.adarho)*u**2
                                         for au,u in zip(self.accum_update[1:], update)]
                # handle dictionary separately
                gL = grad[0] #is a dict
                agL = self.accum_grad[0] #is a matrix
                auL = self.accum_update[0]
                uL = gL.copy() #is a dict
                for j in gL.iterkeys():
                    agL[:,j] = self.adarho*agL[:,j] + (1-self.adarho)*gL[j]**2
                    uL[j] = -np.sqrt(auL[:,j] + self.epsilon) / np.sqrt(agL[:,j] + self.epsilon) * gL[j]
                    auL[:,j] = self.adarho*auL[:,j] + (1-self.adarho)*uL[j]**2
                update = [uL] + update
                scale = -1 #no scaling here

            # update params
            self.model.updateParams(scale,update,log=False)

            self.costt.append(cost)
            if self.it%1 == 0:
                print "Iter %d : Cost=%.4f, ExpCost=%.4f."%(self.it,cost,self.expcost[-1])
            
