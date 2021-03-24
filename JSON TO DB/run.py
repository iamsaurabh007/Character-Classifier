#import threading
import multiprocessing
#import queue
from multiprocessing import Queue
#from src.services.train import train_context
#from src.services.get_corpus import corpus_generator, get_chunk
from utils import generator,put_json_to_database
from config import  PROCESS,JSONPATH,DBNAME
import time
import os
import pymongo
from pymongo import MongoClient


train_queue = Queue() #queue.Queue()
chuck_count = 0
client = MongoClient()
myclient   = pymongo.MongoClient('mongodb://localhost:27017')
mydb       = myclient[DBNAME]
collection = mydb.chardataset

def train_chunk():
    while True:
        
        img=train_queue.get(block=True)
        #print('1')
        put_json_to_database(img,JSONPATH,collection)
        
def start_threads(thread_count):
    threads = []
    for t in range(thread_count):
        threads.append(multiprocessing.Process(target=train_chunk))
        threads[-1].start()


if __name__ == '__main__':

    para = generator(JSONPATH)
    start_threads(PROCESS)     
    while True :
        p=False
        while train_queue.qsize() < PROCESS :
            try:
                im = para.__next__()
                train_queue.put(im)
                #time.sleep(.1)
            except StopIteration:
                p=True
                break
            #time.sleep(1)
            #print('chunks in queue: {}'.format(train_queue.qsize()))
            
        time.sleep(0.05)
        if p:
            break
    print('DB TRANSFER finished')
