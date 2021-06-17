import numpy as np
import sys
def read_frequent_time(fp):
        label = []
        indx = []
        cntx = []
        idx = 0
        stop=0
	temp = fp.readline().strip()
        if len(temp)<1:
                return (label, indx, cntx, 1)
        temp = temp.split(" ")
        print temp[0]
        checktime = temp[0][4:6]
        label.append(int(temp[1]))
        n = len(temp)-2
        ind = np.zeros( n, dtype = np.int32)
        cnt = np.zeros( n, dtype = np.int32)
        for i in range(n):
                ind[i] = int(temp[i+2].split(":")[0])
                cnt[i] = int(temp[i+2].split(":")[1])
        indx.append(ind)
        cntx.append(cnt)
        while True:
                temp = fp.readline().strip()
                if len(temp)<1:
                        stop=1
                        break
                temp = temp.split(" ")
                time = temp[0][4:6]
                if time!=checktime:
                        print temp[0]
                        return (label, indx, cntx, 0)
                label.append(int(temp[1]))
                n = len(temp)-2
                ind = np.zeros( n, dtype = np.int32)
                cnt = np.zeros( n, dtype = np.int32)
                for i in range(n):
                        ind[i] = int(temp[i+2].split(":")[0])
                        cnt[i] = int(temp[i+2].split(":")[1])
                indx.append(ind)
                cntx.append(cnt)
        return (label, indx, cntx, stop)
def split(indx, cntx, mini):
	batchsize = len(indx)
	for i in range(1):
		fp1 = open("part1_"+str(mini)+".txt", 'w')
		fp2 = open("part2_"+str(mini)+".txt", 'w')
		for d in range(batchsize):
			ind = indx[d]
			cnt = cntx[d]
			n=len(cnt)
			p1 = np.random.choice(n, int(n/5), replace=False)
			if len(p1)<1:
				continue
			fp1.write(str(n-len(p1))+" ")
			fp2.write(str(len(p1))+" ")
			for j in range(n):
				if j not in p1:
					fp1.write(str(ind[j])+":"+str(cnt[j])+" ")
				else:
					fp2.write(str(ind[j])+":"+str(cnt[j])+" ")
			fp1.write("\n")
			fp2.write("\n")
if __name__ == '__main__':
	fin = sys.argv[1]
	fp = open(fin, 'r')
	mini=0
	while True:
		mini+=1
		(label, indx, cntx, stop) = read_frequent_time(fp)
		if stop==1:
			break
		split(indx, cntx, mini)
