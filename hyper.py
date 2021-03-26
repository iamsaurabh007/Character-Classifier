import run

if __name__ =='__main__':
    for x,i in enumerate([0.0001]):
        for y,j in enumerate([32,16]):
            run.RUN(i,j)
            print("RUNNING ON ITERATION",x,y)
