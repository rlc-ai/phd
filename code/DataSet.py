import numpy as np


class DataSet:
    """
    Generate simple datasets
    """
    def generate_circles_set(self)->np.array:
        theta=np.linspace(0, 2*np.pi, num=200)
        eta=[2*i*np.pi/3 for i in range(3)]
        r0=0.8
        r1=0.1

        X=[]
        for e in eta:
            x0=r0*np.cos(e)
            y0=r0*np.sin(e)
            #print(x0,y0)
            for t in theta:
                X.append([x0+r1*np.cos(t),y0+r1*np.sin(t)])
        X=np.array(X)
        return X
    
    
    def generate_circle(self)->np.array:
        theta=np.linspace(0, 2*np.pi, num=200)
        r0=0.8

        X=[]
        for t in theta:
            X.append([r0*np.cos(t),r0*np.sin(t)])
        X=np.array(X)
        return X
    