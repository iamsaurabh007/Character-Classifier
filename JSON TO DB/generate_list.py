import config
import utils
import pymongo
from config import DBNAME,symbols

if __name__ == '__main__':
    myclient   = pymongo.MongoClient('mongodb://localhost:27017')
    mydb       = myclient[DBNAME]
    collection = mydb.chardata
    train_ls_grid=utils.sampledds(collection,5000)
    valid_ls_grid=utils.sampledds(collection,500)
    utils.list_to_csv(train_ls_grid,"train_grid_imgs")
    utils.list_to_csv(valid_ls_grid,"valid_grid_imgs")

