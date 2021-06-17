import numpy
import sys
import csv
import nltk
from nltk.corpus import stopwords
if __name__ ==  '__main__':
	filein = sys.argv[1]
	folder = sys.argv[2]
	filewe = sys.argv[3]
	we = {}
	stop_words = set(stopwords.words('english'))
	with open(filewe, 'r') as fp:
		while True:
			temp = fp.readline().strip()
			if len(temp)<1:
				break
			temp = temp.split(' ')
			we[temp[0]] = ' '.join(temp[1:])
	vocab_title = {}
	vocab_full = {}
	num_doc =0
	fpt = open('%s\\ag-title.txt'%(folder), 'w')
	fpf = open('%s\\ag-full.txt'%(folder), 'w')
	with open(filein) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for row in csv_reader:
			fpt.write(row[0]+' ')
			fpf.write(row[0]+' ')
			title = row[1].lower()
			des = row[2].lower()
			tt = nltk.word_tokenize(title)
			tf = nltk.word_tokenize(des)
			for w in tt:
				if w not in stop_words and w in we and w.isalpha():
					if w in vocab_title:
						vocab_title[w] +=1
					else:
						vocab_title[w] = 1
					if w in vocab_full:
						vocab_full[w] +=1
					else:
						vocab_full[w] = 1
					fpt.write(w+' ')
					fpf.write(w+' ')
			for w in tf :
				if w not in stop_words and w in we and w.isalpha():
					if w in vocab_full:
						vocab_full[w] +=1
					else:
						vocab_full[w] =1
					fpf.write(w+' ')	
			fpt.write('\n')
			fpf.write('\n')
	fpt.close()
	fpf.close()
	with open('%s\\vocab-title.txt'%(folder), 'w') as fp:
		for w in vocab_title:
			if vocab_title[w]>2:
				fp.write(w+'\n')
	with open('%s\\vocab-full.txt'%(folder), 'w') as fp:
		for w in vocab_full:
			if vocab_full[w]>2:
				fp.write(w+'\n')
					