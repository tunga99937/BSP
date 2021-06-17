import numpy as np
from sklearn.model_selection import train_test_split
import sys, time

def split_classes(batch_size, train_path):
    f = []
    for i in range(6): # Modify the number of class
        tmp = open("class-" + str(i) + ".txt", "a") # Modify index i here  
        f.append(tmp)
    
    current_docs = [[], [], [], [], [], []] # Modify the number of class
    total_mini_batch = [0, 0, 0, 0, 0, 0] # Modify the number of class
    
    with open(train_path) as f_train:
        for i, doc in enumerate(f_train):
          
            # print ("Processing doc number " + str(i + 1))
            content = doc.strip().split(" ")
            class_id = int(content[1])   # Modify index of content and value of subtraction (ie. 1 is starting index of class)
            current_docs[class_id].append(' '.join(content[2:])) # Modify index of content    
            if len(current_docs[class_id]) == batch_size:
                total_mini_batch[class_id] += 1
                for doc in current_docs[class_id]:
                    f[class_id].write('%s\n' %doc)
                current_docs[class_id] = []
    
    for i in range(6): # Modify the number of class
        f[i].close()
    return total_mini_batch


def split_holdout(batch_size, datapath=None):
    # print('='*50)
    filenames = ['class-0.txt', 'class-1.txt', 'class-2.txt', 'class-3.txt', 'class-4.txt', 'class-5.txt'] # Modifing here for order of class in training data
    for i, filename in enumerate(filenames):
        print('Processing '+ filename)
        docs = []
        with open('./' + filename, 'r') as f:
            for line in f:
                docs.append(line.strip())
        
        # Split data in each class to data and holdout with holdout's size is batch_size
        data, holdout = train_test_split(docs, test_size=batch_size, random_state=42)
        print("length of data is ", len(data))
        print("length of hold-out is ", len(holdout))
        # Write holdout file
        with open('./holdout-'+str(i+1)+'.txt', 'w') as fwrite:
            for doc in holdout:
                fwrite.write('%s\n' %doc)

        # Write data in each class
        with open('./splited_class-'+str(i+1)+'.txt', 'w') as fwrite:
            for doc in data:
                fwrite.write('%s\n' %doc)
        print()
 


def concatenate():
    filenames = ['splited_class-4.txt', 'splited_class-5.txt', 'splited_class-6.txt', 'splited_class-1.txt', 'splited_class-2.txt', 'splited_class-3.txt'] # Modifing here for order of class in training data
    with open('./all-splited-train.txt', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

def write_setting(num_topic, batch_size, str_number_class):
    f_setting = open('setting.txt')
    f_new_setting = open('new_setting.txt', 'w')
    lines = f_setting.readlines()
    for line in lines:
        line = line.strip()
        name = line.split(':')[0]
        if name == 'num_topic':
            f_new_setting.write('num_topic: %s\n' % num_topic)
        elif name == 'batch_size':
            f_new_setting.write('batch_size: %s\n'  % batch_size)
        elif name == 'class':
            f_new_setting.write('class: %s\n' % str_number_class)
        else:
            f_new_setting.write(line + '\n')
    f_setting.close()
    f_new_setting.close()

def main():
    num_topic = int(sys.argv[1])    # 100   
    batch_size = int(sys.argv[2])   # 2000
    train_path = './train_nonstem.txt'
    t1 = time.time()
    total_mini_batch = split_classes(batch_size, train_path)
    str_number_class = ''
    for i, num in enumerate(total_mini_batch):
        print ("Class " + str(i) + ": " + str(num) + " minibatches")
        str_number_class += str(num-1) + ' '

    print ('Write setting file...')
    write_setting(num_topic, batch_size, str_number_class)
    print('='*50)
    print('Split data in all class into two sets...')
    split_holdout(batch_size=batch_size)
    print('='*50)
    print ('Concatenate all training file...')
    concatenate()
    print ('\n')
    t2 = time.time()
    print ('Total time process: ' + str(t2-t1))

if __name__ == '__main__':
    main()
    
