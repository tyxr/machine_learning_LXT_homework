
import random
import re
import numpy as np
def sign(x):
    if x>= 0:
        return 1
    else:
        return -1
def load_data():
    graph ={}
    with open('./pla.dat') as f:
        lines = f.readlines()

        
    lines = map(lambda s: re.sub('\s+',' ',str(s.strip('\r\n'))).strip(),lines)    
    lines = map(lambda s: s.split(' '),lines)
    
    
    lines = np.array(list(lines))
    lines.reshape(400,5)
    one = np.ones((len(lines)))
    lines = np.c_[one,lines]
    lines = lines.astype('float64')
    print(lines[0])
    return lines
def pla(dataset):
    W=np.zeros(5)#initial all weight with 1
    count=0

    while True:
        count+=1
        iscompleted=True
        for i in range(len(dataset)):
            
            X=dataset[i][:-1]
            Y=np.dot(W,X)#matrix multiply
            if sign(Y)==sign(dataset[i][-1]):
                pass
            else:
                count+=1
                iscompleted=False
                W=W+(dataset[i][-1])*np.array(X)
                
        if iscompleted:
            break
    print("final W is :",W)
    print("count is :",count)
    return W

def main():
    dataset = load_data()
    pla(dataset)
    
if __name__ == '__main__':
    main()
