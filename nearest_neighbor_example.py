import numpy as np
from utils import find_nearest_neighbor
'''
training shape(n,m)
n:input_size
m:feature_dim

if initial shape and train shape different
raise exception

search shape(n,m),k
n:input_size
m:feature_dim
k:number of nearest neighbor
'''

x = np.random.random((10,20))
a= find_nearest_neighbor(x)
x = np.random.random((10,20))
a.train(x)
print(a.get_length())
print(len(a.get_data()))
query = np.random.random((1,20))
dists, array = a.search(query)

print(dists.shape,array.shape)
