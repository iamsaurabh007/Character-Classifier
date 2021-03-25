import run

if __name__ =='__main__':
    for x,i in enumerate([0.1,0.01,0.001,0.0001,0.00001]):
        for y,j in enumerate([1,32,64]):
            run.RUN(i,j)
            print("RUNNING ON ITERATION",x,y)
