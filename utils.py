from sklearn.neighbors import KDTree
import numpy as np

class find_nearest_neighbor():
    '''This is an interface use KDTree from sklearn find nearest neighbor
        x:shape(n,m)
        n:input_size
        m:feature_dim
    '''

    def __init__(self, x=[], metric = 'euclidean'):
        self.n = len(x)
        if self.n!=0:
            self._tree = KDTree(x, leaf_size=2, metric=metric)
            self.data = x

    def train(self, x, metric = 'euclidean'):
        if self.n!= 0:
            if self._tree.data.shape[1:]==x.shape[1:]:
                self.data = np.concatenate([self.data,x])
                x = self.data
                self.n = len(x)
            else:
                raise('training set shape is different')
        self._tree = KDTree(x, leaf_size=2, metric=metric)

    def search(self,query,k=5):
        '''search k nearest neighbor
            return type
            dist shape(n,k)
            array shape(n,k,m)
            n: input_size
            k: number nearest neighbor
            m: feature_dim
        '''
        if self.n ==0:
            raise('tree didn\'t train')
        n,m = query.shape
        dists, inds = self._tree.query(query,k=k)
        return dists, self.data[inds]

    def get_length(self):
        print(self.n)

    def get_data(self):
        return self._tree.data
