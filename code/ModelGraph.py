import tensorflow as tf
import numpy as np
import copy


class ModelGraph:
    """
    The class extract the graph from a sequential keras model.
    TODO: check and extend for general model
    """
    def __init__(self, model: tf.keras.models.Model):
        layer_sizes=[]
        for layer in model.layers:
            if type(layer.output_shape)==list:
                layer_sizes.append(layer.output_shape[0][-1])
            else:
                layer_sizes.append(layer.output_shape[-1])
        self.V=[] # vertexes
        self.E=[] # edges
        self.V.append(list(range(layer_sizes[0]))) # neuroni di input
        for l in layer_sizes[1:]:
            self.V.append([v+max(self.V[-1])+1 for v in list(range(l))])
            for v in self.V[-2]:
                for w in self.V[-1]:
                    self.E.append([v,w])
            self.V.append([v+max(self.V[-1])+1 for v in list(range(l))])
            for i in range(len(self.V[-1])):
                self.E.append([self.V[-2][i],self.V[-1][i]])
        self.w=[]
        for layer in model.layers:
            self.w.append(copy.deepcopy(layer.get_weights()))
    
    def predict(self, X:np.array) ->list:
        return [self.compute_edges(x)[-2:] for x in X]
    
    def error(self, X)->float:
        return np.linalg.norm(X-self.predict(X),axis=1)

    def compute_edges(self, point:np.array)->list:
        edges=[]
        p=point
        for i in range(1,len(self.w)):
            edges.extend((self.w[i][0]*(p.reshape(-1,1))).reshape(-1).tolist())
            r=np.dot([p],self.w[i][0])[0].tolist()
            q=np.tanh(np.dot([p],self.w[i][0])[0]+self.w[i][1])
            edges.extend(q.reshape(-1).tolist())
            p=q
        return edges
    
    def set_adj_entry(self, A:np.array, index:int, value:int):
        i,j=self.E[index]
        A[i,j]=value
        A[j,i]=value
        
    def compute_adiacent_matrix(self,point:np.array)->np.array:
        edges=self.compute_edges(point)
        max_vertex_index=max([max(v) for v in self.V])
        A=np.zeros((max_vertex_index+1,max_vertex_index+1))
        for i in range(len(edges)):
            self.set_adj_entry(A,i,edges[i])
            
        return A
    
    def compute_energy(self,point:np.array)->float:
        """
        Compute the energy of the graph (point based)
        """
        return np.sum(np.absolute(np.real(np.linalg.eig(self.compute_adiacent_matrix(point))[0])))