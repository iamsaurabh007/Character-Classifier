import pymongo
from pymongo import MongoClient

client = MongoClient()
myclient   = pymongo.MongoClient('mongodb://localhost:27017')
mydb = myclient['db']
images=mydb['chardataset']
for i,j in enumerate(images.find()):
    print(j)
    if i==8:
        break
    