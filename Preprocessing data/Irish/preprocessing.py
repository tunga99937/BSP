import numpy
import sys
import csv
import nltk
import datetime
from nltk.corpus import stopwords
import sys
reload(sys)
sys.setdefaultencoding('utf8')
if __name__ ==  '__main__':
	filein = sys.argv[1]
	folder = sys.argv[2]
	filewe = sys.argv[3]
	we = {}
	print('Reading wordembedding')
	stop_words = set(stopwords.words('english'))
	with open(filewe, 'r') as fp:
		while True:
			temp = fp.readline().strip()
			if len(temp)<1:
				break
			temp = temp.split(' ')
			we[temp[0]] = ' '.join(temp[1:])
	vocab = {}
	label = {}
	idx = 0
	num_doc =0
	print(len(we))
	fpt = open('%s\\irish-raw.txt'%(folder), 'w')
	print('Preprocessing')
	with open(filein) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		next(csv_reader)
		for row in csv_reader:
			num_doc+=1
			
			fpt.write(row[0] + ' ')
			strl = row[1]
			if strl in label:
				fpt.write(str(label[strl])+' ')
			else:
				idx+=1
				label[strl]=idx
				fpt.write(str(idx)+' ')
			title = row[2].lower()
			tt = nltk.word_tokenize(title)
			for w in tt:
				if w not in stop_words and w in we and w.isalpha():
					if w in vocab:
						vocab[w] +=1
					else:
						vocab[w] = 1
					fpt.write(w+' ')	
			fpt.write('\n')
	fpt.close()
	print(label)
	print(num_doc)
	with open('%s\\vocab_nonstem.txt'%(folder), 'w') as fp:
		for w in vocab:
			if vocab[w]>2:
				fp.write(w+'\n')
					