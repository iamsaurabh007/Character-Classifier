import json
from os import listdir
from os.path import isfile, join
import random
import numpy as np
import config


def get_images_list(mypath):
    #Currently jpeg implementation only
    onlyfiles = [f[:-5] for f in listdir(mypath) if isfile(join(mypath, f))]
    print("Images Available : {} ".format(len(onlyfiles))) 
    return onlyfiles

def generator(path):
    ls=get_images_list(path)
    for i in ls:
        yield i
        
def put_json_to_database(img,dirpath,client_collect):   #################
    with open(dirpath+img+'.json') as f:
        p = json.load(f)
   
    p['_id']=p.pop('image_id')
    client_collect.insert_one(p)


def sampledds(client,length):
    ls=[]
    characters=config.symbols
    #characters=['A','a','P','O']
    s=length//len(characters)
    for ch in characters:
        myquery = { "character": ch }
        #myquery={"$and":[{"character":ch}]}
        p=[]
        for i in client.find(myquery):
            p.append(i['_id'])
        q=min(s,len(p))
        print(ch," : ",q)
        random.shuffle(p)
        ls.extend(p[:q])
    print("Length of Data Samples Returned with even distribution is: ", len(ls))
    random.shuffle(ls)
    return ls

def list_to_csv(ls,filename):
    np.savetxt(filename+".csv",ls,delimiter =", ", fmt ='% s')s
