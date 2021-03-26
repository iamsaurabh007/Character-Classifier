import pandas
import pymongo


if __name__ =='__main__':
    client = pymongo.MongoClient()
    db=client['db']
    images=db['chardata']
    myquery={"font_size":{"$gt":5},"background":{ '$nin': [ "pexels-photo-479453.jpeg", "jude-beck-FQaFVRGJ9uk-unsplash.jpg"] } }
    data=[]
    for i in images.find(myquery):
        data.append(i)
    df=pd.DataFrame(data)
    df.to_csv("/home/ubuntu/data/ocr"+"/datachar.csv",index=False)
    print("CSV CREATION DONE")
