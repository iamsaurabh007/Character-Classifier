import run

if __name__ =='__main__':
    for x,i in enumerate([0.001]):
        for y,j in enumerate([16,32]):
            run.RUN(i,j)
            print("RUNNING ON ITERATION",x,y)
