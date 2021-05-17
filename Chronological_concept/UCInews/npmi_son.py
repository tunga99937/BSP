import numpy as np 
import itertools as it
from collections import defaultdict
import sys, os, time,glob

freq_i_j = defaultdict(float)

def write_result(LD, file_name):
    f = open(file_name, 'a')
    f.write('%f,'%(LD))
    f.close()

def read_beta(file_path):
	data = np.loadtxt( file_path , dtype = "int32")
	return data

def read_data(filename):
	wordinds = list()
	wordcnts = list()
	fp = open(filename, 'r')
	while True:
		line = fp.readline().strip() # string '6 2:1 571:1 9:1 569:1 123:2 572:1'
		# check end of file
		if len(line) < 1:
			break
		terms = str.split(line) # ['2:1', '571:1', '9:1', '569:1', '123:2', '572:1']
		terms = terms[2:]
		doc_length = len(terms)
		inds = np.zeros(doc_length, dtype = np.int32)
		cnts = np.zeros(doc_length, dtype = np.int32)
		for j in range(0, doc_length):
			term_count = terms[j].split(':') # such as ['2', '1']
			inds[j] = int(term_count[0])
			cnts[j] = int(term_count[1])
		wordinds.append(inds) # A list D elements, each element is an array such as array([  2, 571,   9, 569, 123, 572])
		wordcnts.append(cnts)
	fp.close()
	return (wordinds, wordcnts)

# =============================================================================
# def freq_per_word(wordinds):
#     p_i = defaultdict(float)
#     for doc in wordinds:
#         for w in doc:
#             p_i[w] += 1
#     num_docs = len(wordinds)
#     for idx in p_i:
#         p_i[idx] /= num_docs
#     return p_i
#         
# def freq_per_couple(beta, wordinds):
#     def get_all_couple(beta):
#         all_couples = {}
#         for beta_i in beta:
#             for couple in list(it.combinations(beta_i, 2)):
#                 all_couples[couple] = 0
#         return all_couples
#                 
#     all_couples = get_all_couple(beta)
#     p_i_j = defaultdict(float)
#     for doc in wordinds:
#         for couple in all_couples:
#             if couple[0] in doc and couple[1] in doc:
#                 p_i_j[couple] += 1
#     for couple in p_i_j:
#         p_i_j[couple] /= len(wordinds)
#     return p_i_j
# =============================================================================

def compute_inverse_index(wordinds):
    inv_idx = defaultdict(list)
    for doc_id, doc in enumerate(wordinds):
        for w_id in doc:
            inv_idx[w_id].append(doc_id)
    return inv_idx

def npmi_beta_k(index_top_words, num_docs, inv_idx):
	num_top = len(index_top_words)
	num_couple = num_top * (num_top - 1) / 2
	npmi = 0
	for couple in list(it.combinations(index_top_words, 2)):
		w_1 = couple[0]
		w_2 = couple[1]
# =============================================================================
# 		if p_i[w_1] == 0:
#             for doc in wordinds:
#                 if w_1 in doc:
#                     p_i[w_1] += 1
# 		 	p_i[w_1] /= len(wordinds)
# 		if p_i[w_2] == 0:
# 		 	for doc in wordinds:
# 		 		if w_2 in doc:
# 		 			p_i[w_2] += 1
# 		 	p_i[w_2] /= len(wordinds)
# 		if p_i_j[couple] == 0:
# 			for doc in wordinds:
# 				if w_1 in doc and w_2 in doc:
# 					p_i_j[couple] += 1
# 			p_i_j[couple] /= len(wordinds)
# =============================================================================
		freq_w_1 = len(inv_idx[w_1])
		freq_w_2 = len(inv_idx[w_2])
		epsi =0.01
		if couple not in freq_i_j:
			freq_i_j[couple] = len(set(inv_idx[w_1]).intersection(inv_idx[w_2]))
		if freq_w_1 ==0:
			freq_w_1=epsi
		if freq_w_2 ==0:
			freq_w_2=epsi
		if freq_i_j[couple] == 0:
			npmi_couple = np.log(1.0 * freq_w_1 * freq_w_2 / num_docs**2) / np.log(1.0 * epsi / num_docs) - 1
		else:
			npmi_couple = np.log(1.0 * freq_w_1 * freq_w_2 / num_docs**2) / np.log(1.0 * freq_i_j[couple] / num_docs) - 1
		npmi += npmi_couple
	return npmi / num_couple

def npmi(beta, num_docs, inv_idx):
	mean_npmi = 0.0
	for k, beta_k in enumerate(beta):
		mean_npmi += npmi_beta_k(beta_k, num_docs, inv_idx)
	return mean_npmi / beta.shape[0]

def main():
	filetrain = sys.argv[1]
	folder_models=sys.argv[2]
	num_minibatch = int(sys.argv[3])
	wordinds, wordcnts = read_data(filetrain)
	num_docs = len(wordinds)
	print (num_docs)
	print ("Computing inverse index...")
	inv_idx = compute_inverse_index(wordinds)
	for model in glob.glob('%s/*'%(folder_models)):
		t1 = time.time()
		print(model)
		npmi_file = '/'.join([model, 'npmi.csv'])
		for i in range(1,num_minibatch+1):
			try:
				method = model.split('-')[-1]
			#file_path = model + '/top20_1_' + str(i) + '.dat'
				file_path = model+'/list_tops_'+str(i)+'.dat'
				#if method in ['W2V','WN']:
				#	print model
				#	file_path = model + '/top20_1_' + str(i) + '.dat'
				#else:
				#	file_path = model + '/list_tops_' + str(i) + '.dat'
				beta_i = read_beta(file_path)
				npmi_i = npmi(beta_i, num_docs, inv_idx)
				print ('Minibatch ' + str(i+1) + " : " + str(npmi_i))
				write_result(npmi_i, npmi_file)
			except:
				print(model)
				continue
		t2 = time.time()
		print ('Total time calculate - ' + filetrain + ' - : ' + str(t2 - t1))

if __name__ == '__main__':
	main()




