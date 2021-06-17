import numpy
import sys
import csv
import nltk
from nltk.corpus import stopwords
if __name__ ==  '__main__':
	filein = sys.argv[1]
	filevocab = sys.argv[2]
	folder = sys.argv[3]
	filewe = sys.argv[4]
	we = {}
	with open(filewe, 'r') as fp:
		while True:
			temp = fp.readline().strip()
			if len(temp)<1:
				break
			temp = temp.split(' ')
			we[temp[0]] = ' '.join(temp[1:])
	vocab = {}
	index = 0
	fpp = open('%s\\prior.glove.200d_nonstem.txt'%(folder), 'w')
	with open(filevocab, 'r') as fp:
		while True:
			temp = fp.readline().strip()
			if len(temp)<1:
				break
			vocab[temp] = index
			fpp.write(we[temp]+'\n')
			index+=1
	fpp.close()
	fpt = open('%s\\train_nonstem.txt'%(folder), 'w')
	n = 0
	with open(filein, 'r') as fp:
		while True:
			temp = fp.readline().strip()
			if len(temp)<1:
				break
			temp = temp.split(' ')
			ts = temp[0]
			label  = temp[1]
			temp = temp[2:]
			doc = {}
			for w in temp:
				if w in vocab:
					if w in doc:
						doc[w]+=1
					else:
						doc[w]=1
			if len(doc)>1:
				fpt.write('%s %s '%(ts, label))
				for w in doc:
					fpt.write('%s:%s '%(vocab[w], doc[w]))
				fpt.write('\n')
	fpt.close()

					