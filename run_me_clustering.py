import numpy as np
import matplotlib.pyplot as plt
import util
from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn import metrics

#Load the data. imgs is an array of shape (N,8,8) where each 8x8 array
#corresponds to an image. imgs_vectors has shape (N,64) where each row
#corresponds to a single-long-vector respresentation of the corresponding image.
img_size = (8,8)
imgs, imgs_vectors  = util.loadDataQ1()

#You should use the data in imgs_vectors to learn a clustering model.
#This starter code uses random clusters.
K  = 10
N  = imgs_vectors.shape[0]
Zs = np.random.random_integers(1,K,N)

random_state = 100

## test different models
"""
y_pred = KMeans(n_clusters=K, random_state=random_state).fit_predict(imgs_vectors)
two_means = cluster.MiniBatchKMeans(n_clusters=K,random_state=random_state).fit_predict(imgs_vectors)
spectral = cluster.SpectralClustering(n_clusters=K,eigen_solver='arpack', affinity="nearest_neighbors", random_state=random_state).fit_predict(imgs_vectors)

##print imgs[0]
#print y_pred
#print two_means
#print spectral


print "kmeans"
for K in range(2,21,1):
	labels = KMeans(n_clusters=K, random_state=random_state).fit(imgs_vectors).labels_
	score = metrics.silhouette_score(imgs_vectors, labels, metric='euclidean')
	print str(K) +" : ", score

print "minibatch"
for K in range(2,21,1):
	labels = cluster.MiniBatchKMeans(n_clusters=K,random_state=random_state).fit(imgs_vectors).labels_
	score = metrics.silhouette_score(imgs_vectors, labels, metric='euclidean')
	print str(K) +" : ", score
"""

print "spectral  - scoring model has to start at K=2"
bestk = {}
bestkchange = {}
prevscore = 0
for k in range(2,21,1):
	labels = cluster.SpectralClustering(n_clusters=k,eigen_solver='arpack', affinity="nearest_neighbors", random_state=random_state).fit(imgs_vectors).labels_
	score = metrics.silhouette_score(imgs_vectors, labels, metric='euclidean')
	if k==2:
		prevscore = score
	bestkchange[k]=score - prevscore
	bestk[k] = score
	prevscore = score
	print str(k) +" : ", score

##normalize and find the bestk
normalizek = {}
number1 = sorted(bestk.iteritems(), key=lambda x:-x[1])[:1][0]
print number1
for key in bestk.keys():
	if key !=20:
		normalizek[key] =(1+bestkchange[key]- bestkchange[key+1]) /(1+ number1[1]-bestk[key])
		print  str(key) + " normalize score: ", (1+bestkchange[key]- bestkchange[key+1]) /(1+ number1[1]-bestk[key])


#print normalizek
normalizek1 = sorted(normalizek.iteritems(), key=lambda x:-x[1])[:1][0][0]
K = normalizek1
print K

##output results models
Zs = cluster.SpectralClustering(n_clusters=K,eigen_solver='arpack', affinity="nearest_neighbors", random_state=random_state).fit_predict(imgs_vectors)

## print number count of each cluster
for k in np.unique(Zs):
	print str(k) +" num of it ", len([ks for ks in Zs if ks==k])





#The code below shows how to plot examples from clusters as an image array
for k in np.unique(Zs):
	count = len([ks for ks in Zs if ks==k])
	plt.figure(k)
	if np.sum(Zs==k)>0:
  	  util.plot_img_array(imgs_vectors[Zs==k,:], img_size,grey=True)
  	plt.suptitle("Cluster Exmplars %d/%d - num of this: %d"%(k,K, count))
#plt.show()

#The code below shows how to compute and plot cluster centers as an image array
centers = np.zeros((len(np.unique(Zs)),64))
plt.figure(1)
i=0
for k in np.unique(Zs):
    centers[i,:] = np.mean(imgs_vectors[Zs==k,:],axis=0)
    i=i+1
util.plot_img_array(centers, img_size,grey=True)
plt.suptitle("Cluster Centers (K=%d)"%K)
#plt.show()
