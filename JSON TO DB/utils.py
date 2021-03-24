import json


def get_images_list(mypath):
    #Currently jpeg implementation only
    onlyfiles = [f[:-5] for f in listdir(mypath) if isfile(join(mypath, f))]
    print("Images Available : {} ".format(len(onlyfiles))) 
    return onlyfiles

def generator(path):
    ls=get_images_list(path)
    for i in ls:
        yield i
        
def put_json_to_database(img,dirpath,clientcollect):   #################
    with open(dirpath+img+'.json') as f:
        data = json.load(f)
    p=data['image']
    p['_id']=p.pop('image_id')
    client_collect.insert_one(p)
