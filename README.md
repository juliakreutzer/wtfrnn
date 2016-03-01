# wtfrnn
Weak Tree Feedback Recursive Neural Networks for Sentiment Classification

The RNTN [1] model and its variation are implemented on the basis of course material [2] of the Stanford course "CS224d: Deep Learning for Natural Language Processing" by Richard Socher. The course material includes the training, development and test set for the Sentiment Treebank in Penn Treebank bracketing format, and the code to load and process these trees.
Gradients, feed-forward and backpropagation passes had to be completed. 

On top of this model, I built a number of modifications that allowed to further investigate the single components of this model. These are:
1. training word2vec word embeddings with gensim to initialize the lookup table of the RNTN model (`w2v.py`, training option `--w2v`)
2. transform sentences into a general tree format to train the RNTN model on un-parsed sentences (`fromTextToFullTree.py`)
3. training models without phrase-based annotations (training option `--partial`)

To train a model, use `run.sh` with the parameters listed in `runNNet.py`. To test a model, use `test.sh` accordingly. Models are stored as pickle objects. Before running the code, execute `setup.sh` to download the data and create directories and build trees. The input to the trained model (full trees or dependency trees) has to be defined in `runNNet.py`.


[1] Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank, Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Chris Manning, Andrew Ng and Chris Potts. 
Conference on Empirical Methods in Natural Language Processing (EMNLP 2013)

[2] http://cs224d.stanford.edu/assignment3/index.html
