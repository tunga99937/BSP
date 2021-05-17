import numpy as np
import sys, os, shutil
import model_LDA
import utilities
from shutil import copyfile
import pickle

#all_dataset_dir = 'dataset/6-statictarget-datasets'
all_dataset_dir = 'dataset'
#all_dataset = {'20newgroups', 'Grolier', 'NYtimes', 'TMN', 'TMNtitle', 'Twitter', 'Yahoo', 'Agnews', 'Agnews-title'}
all_model_folder = 'models'

def copy_results(src, dst):
    flag = 1
    if os.path.exists(dst):
        flag = 0
    if flag == 1:
        shutil.copytree(src, dst, symlinks=False, ignore=None)

def main():
    # if(len(sys.argv) != 8):
    #     print('Usage: python run.py [dataset_name] [times] [type_model] [rate] [weight] [iters] [start]')
    #     exit()
    
    if(len(sys.argv) != 7):
        print('Usage: python run.py [times] [type_model] [rate] [weight] [iters] [start]')
        exit()

    times = sys.argv[1]
    type_model = sys.argv[2]
    rate = float(sys.argv[3])
    weight = float(sys.argv[4])
    iters = int(sys.argv[5])
    mini_batch = int(sys.argv[6])

    # dataset_name = 'irishtime-unsupervise'
    # mean = sys.argv[1]
    # sigma = sys.argv[2]
    # drop_rate = float(sys.argv[3])
    # times = sys.argv[4]
    # type_model = sys.argv[5]
    # epochs = int(sys.argv[6])
    # iters = int(sys.argv[7])
    # mini_batch = int(sys.argv[8])

    # dataset_dir = '/'.join([all_dataset_dir, dataset_name])
    # train_file = '/'.join([dataset_dir, 'train.txt'])
    # setting_file = '/'.join([dataset_dir, 'setting.txt'])
    # vocab_file = '/'.join([dataset_dir, 'vocab.txt'])
    # prior_path = '/'.join([dataset_dir, 'prior.glove.200d.txt'])
    #dataset_dir = '/'.join([all_dataset_dir, dataset_name])
    train_file = '/'.join([all_dataset_dir, 'train.txt'])
    setting_file = '/'.join([all_dataset_dir, 'setting.txt'])
    vocab_file = '/'.join([all_dataset_dir, 'vocab.txt'])
    prior_path = '/'.join([all_dataset_dir, 'prior.glove.200d.txt'])

    if(type_model == 'B'):
        model_name = 'bernoulli'
    elif(type_model == 'S'):
        model_name = 'standard'
    elif(type_model == 'Z'):
        model_name = 'init_zero'
    else:
        print('Unknown type model!')
        exit()
    
    # mean_folder = 'mean' + mean

    model_folder = '/'.join(['models', all_model_folder]) + 'from_batch' + str(mini_batch)
    setting_folder = '/'.join([model_folder, 'new_setting'])
    new_setting_folder = '/'.join([setting_folder, dataset_name])

    if not os.path.exists(new_setting_folder):
        os.makedirs(new_setting_folder)
    
    part_result = '/'.join([model_folder, 'result_' + times])
    GDS_result = '/'.join([part_result, dataset_name + '-' + str(type_model) + '-rate' + str(rate) + '-' + str(weight) + '-' + str(iters)])
    
    if(os.path.exists(GDS_result)):
        shutil.rmtree(GDS_result)
    os.makedirs(GDS_result)

    new_setting_file ='/'.join([new_setting_folder, 'new_setting_file.txt'])

    f_setting = open(setting_file)
    if not os.path.exists(new_setting_file):
        f_new_setting = open(new_setting_file, 'w')
        lines = f_setting.readlines()
        for line in lines:
            line = line.strip()
            name = line.split(':')[0]
            # if(name == 'mean'):
            #     f_new_setting.write('mean: %s\n' % mean)
            # elif(name == 'sigma'):
            #     f_new_setting.write('sigma: %s\n' % sigma)
            # elif(name == 'variance'):
            #     f_new_setting.write('variance: %s\n' % variance)
            # else:
            f_new_setting.write(line + '\n')
        f_setting.close()
        f_new_setting.close()

    print('Reading setting...')
    setting = utilities.read_setting(new_setting_file)
    num_topic = setting['num_topic']
    n_term = setting['n_term']
    batch_size = setting['batch_size']
    n_infer = setting['n_infer']
    learning_rate = setting['learning_rate']
    alpha = setting['alpha']
    # sigma = setting['sigma']
    # mean = setting['mean']
    # variance = setting['variance']

    f_train = open(train_file, 'r')

    
    if(type_model == 'B'):
        MODEL = model_LDA.Model(prior_path, num_topic, n_term, batch_size, n_infer, learning_rate, alpha, rate, 0, weight, iters, mini_batch)
    elif(type_model == 'S'):
        MODEL = model_LDA.Model(prior_path, num_topic, n_term, batch_size, n_infer, learning_rate, alpha, rate, 1, weight, iters, mini_batch)
    elif(type_model == 'Z'):
        MODEL = model_LDA.Model(prior_path, num_topic, n_term, batch_size, n_infer, learning_rate, alpha, rate, 2, weight, iters, mini_batch)
    else:
        print('Unknown type model!')
        exit()
    #MODEL = model_LDA.Model(prior_path, num_topic, n_term, batch_size, n_infer, learning_rate, alpha, weight, iters, mini_batch)
    #wordinds1, wordcnts1, wordinds2, wordcnts2 = utilities.read_data_for_perplex(dataset_dir, num_test)

    
    PPL = []

    perplexity_file = '/'.join([GDS_result, 'perplexities_from_batch' + str(mini_batch) + '.csv'])

    # for i in range(mini_batch):
    #     #(wordinds, wordcnts, stop) = utilities.read_minibatch_list_frequencies(f_train, batch_size)
    #     (label, wordinds, wordcnts, stop) = utilities.read_frequent_time(f_train)
    prev_line = ''
    num_day = 2
    (wordinds, wordcnts, labels, stop, prev_line) = \
		utilities.read_minibatch_frequencies_time_days(f_train, num_day, prev_line)

    while True:
        mini_batch += 1
        print('[MINIBATCH %d]' % mini_batch)
        MODEL.update_pi(wordinds, wordcnts, mini_batch)
        beta_minibatch = MODEL.beta

        if stop == 1:
            break

        (wordinds, wordcnts, labels, stop, prev_line) = \
			utilities.read_minibatch_frequencies_time_days(f_train, num_day, prev_line)	
        
        wordinds1, wordcnts1, wordinds2, wordcnts2 = utilities.split_data_for_perplex(wordinds, wordcnts)

        print ('Compute perplexity...')
        (LD, ld2) = utilities.compute_perplexity(wordinds1, wordcnts1, wordinds2, wordcnts2, num_topic, n_term, n_infer, alpha, beta_minibatch)
        print ('Minibatch ' + str(mini_batch) + ' |=====| PERPLEXITY : ' +  ' sigma' + str(sigma) + ' rate' + str(rate) + ' : ' + str(LD) +  '\n')
        PPL.append(LD)

        utilities.write_perplexities(LD, perplexity_file)

        list_tops_minibatch = utilities.list_top(beta_minibatch, 20)
        list_tops_minibatch_file = '/'.join([GDS_result, 'list_tops_' + str(mini_batch) + '.dat'])
        utilities.write_topic_top(list_tops_minibatch, list_tops_minibatch_file)

    print(PPL)
    name_model = dataset_name + '-' + str(type_model) + '-rate' + str(rate) + '-' + str(weight) + '-' + str(iters)
    model_PPL = []
    model_PPL.append(name_model)
    model_PPL.extend(PPL)
    dataset_all_lpp_file = dataset_name + '.csv'
    utilities.append_perplexities(model_PPL, dataset_all_lpp_file)

    #print beta
    beta_final = MODEL.beta
    beta_file = '/'.join([GDS_result, 'beta_final_' + times + '.txt'])
    utilities.write_topics(beta_final, beta_file)

    #print list_top
    list_tops_final = utilities.list_top(beta_final, 20)
    list_tops_final_file = '/'.join([GDS_result, 'list_tops_final' + times  + '.txt'])
    utilities.write_topic_top(list_tops_final, list_tops_final_file)

    #print top word
    top_word_file = '/'.join([GDS_result, 'top_word.txt'])
    utilities.write_top_word(list_tops_final, vocab_file, top_word_file)

    f_train.close()


if __name__ == '__main__':
    main()
