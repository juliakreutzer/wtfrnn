import sys
import gensim.models
import numpy as np
import tarfile
import codecs

""" 
Training a Word2Vec model on given corpus
Parameters: 1)d_wrd 2)model output file
""" 


if __name__=="__main__":
	#open input files (en + es)
	en_files = ["/home/students/kreutzer/wtfrnn/data/movies/lmdb/aclImdb/all.txt.tok"]
	

	sents = [["UNK"]]
	for f in en_files:
		print "Reading documents from file", f
		infile = codecs.open(f, "r", "utf8")
		for line in infile:
			if len(line)<2:
				continue
			tokenized = line.strip().lower().split() #already tokenized
			sents.append(tokenized)
	print "Read all documents."

	print "Read %d sentences" % len(sents)

	#build gensim model
	windowsize = int(sys.argv[1])
	print "Building the word2vec model with windowsize=%d" % windowsize
	
	model = gensim.models.word2vec.Word2Vec(sents, size=30, window=windowsize, min_count=20, workers=15)

	#persist model
	modelfile = sys.argv[2]
	print "Persisting the model in %s" % modelfile
	model.save(modelfile)
	print "Done."

	print model["enjoy"]
	print model["europe"]
	print model.most_similar(positive=["europe"])
	print model.similarity("film", "movie")

