
import util
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from sklearn.decomposition import dict_learning_online
from sklearn.decomposition import FastICA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import MiniBatchDictionaryLearning


#Set patch size -- change this to set the size of the patches that 
#the de-noising methods will operate over. The input image size is 
#900 rows x 1200 columns. The patch size is also listed as patch 
#height x patch width. The patch height and width must divide into
#the image width and weight evenly.
patch_size = (10,10)

#Load the data. data_noisy and data_clean are arrays with one
#row per image patch. The rows are formed by creating a single
#long vectors of size patch_size[0]*patch_size[1]*3 from the 
#color image patach of size patch_size[0]xpatch_size[1]. img_noisy
#and img_clean are the full clean and noisy imges.
data_noisy, data_clean, img_noisy, img_clean = util.loadDataQ2(patch_size)

print "data vector shape: ", data_noisy.shape
print "img shape: ",img_noisy.shape
h, w = patch_size


#You should learn the dimensionality rediction-based de-noising models 
#on the *noisy* data data_noisy, and then use them to de-noise the noisy data. 
#This process is called reconstruction. Your de-noised data should 
#have the same shape as data_noisy (and data_clean), and should be
#placed in the array data_denoised as seen below. This starter
#code can be thought of as applying the identiity function as the de-noising
#operator.
data_denoised = data_noisy


### testing PCA model
#"""
K=10;
pca        = PCA(n_components=K, whiten=False).fit(data_noisy)
components = pca.components_

code = pca.transform(data_noisy)
patches = pca.inverse_transform(code)
data_denoised =patches
print data_denoised.shape
#"""

### testing Kernal PCA model
"""
K=10;
pca        = KernelPCA(n_components=K, kernel='rbf',fit_inverse_transform=True).fit(data_noisy)
code = pca.transform(data_noisy)
patches = pca.inverse_transform(code)
data_denoised =patches 
print data_denoised.shape
"""

### testing fast ica PCA model
"""
K=10;
pca        = FastICA(n_components=K ).fit(data_noisy)
components = pca.components_
code = pca.transform(data_noisy)
patches = pca.inverse_transform(code)

data_denoised =patches 
print data_denoised.shape
"""

## testing on minibatch  
## do parameter selection over k,alpha and maybe patch size
"""
print "\tnow check best model size"
## loop to find the lowest evaul score to cleanr util.eval_recon(img_clean,img_denoised)
bestK = {}
alpha=1
for i in range(2,10,1):
	step = i**2
	print ""
	print "\tAt step: ", step
	sc         = MiniBatchDictionaryLearning(n_components=step, alpha=alpha, batch_size=10, verbose=True,n_iter=100).fit(data_noisy)
	components = sc.components_  #.reshape((K, h, w))

	code = sc.transform(data_noisy)
	patches = np.dot(code, components)
	data_denoised =patches
	img_denoised  = util.patchToImage(util.vectorsToPatches(data_denoised,patch_size),(900,1200))
	bestK["K:"+str(step)] = util.eval_recon(img_clean,img_denoised)

bestksorted = sorted(bestK.iteritems(), key=lambda x:x[1])
print ""
print bestksorted	

"""

"""
print "\tnow check for alpha changes"
bestK = {}
K = 10
for i in range(-3,3,1):
	step = 10**(i)
	print ""
	print "\tAt step: ", step
	sc         = MiniBatchDictionaryLearning(n_components=K, alpha=step, batch_size=10, verbose=True,n_iter=100).fit(data_noisy)
	components = sc.components_  #.reshape((K, h, w))

	code = sc.transform(data_noisy)
	patches = np.dot(code, components)
	data_denoised =patches
	img_denoised  = util.patchToImage(util.vectorsToPatches(data_denoised,patch_size),(900,1200))
	bestK["a:"+str(step)] = util.eval_recon(img_clean,img_denoised)

bestksorted = sorted(bestK.iteritems(), key=lambda x:x[1])
print ""
print bestksorted	
"""

#"""
#final grid search sweep
bestK = {}
for i in range(2,4,1):
	K = i**2
	for i in range(-3,0,1):
		alpha = 10**(i)
		print ""
		print "\tK: ", K," alpha: ", alpha
		sc         = MiniBatchDictionaryLearning(n_components=K, alpha=alpha, batch_size=10, verbose=True,n_iter=100).fit(data_noisy)
		components = sc.components_  #.reshape((K, h, w))

		code = sc.transform(data_noisy)
		patches = np.dot(code, components)
		data_denoised =patches
		img_denoised  = util.patchToImage(util.vectorsToPatches(data_denoised,patch_size),(900,1200))
		bestK["K:"+str(K)+"_a:"+str(alpha)] = util.eval_recon(img_clean,img_denoised)

bestksorted = sorted(bestK.iteritems(), key=lambda x:x[1])
print ""
print bestksorted	

best_parameters = map(lambda s: s.split(":"), bestksorted[0][0].split("_"))
print best_parameters
#"""

#"""
##run the final model to output
print " now final run for best score, k== 10, alpha =0.01"
K=10;alpha=0.001;

sc         = MiniBatchDictionaryLearning(n_components=K, alpha=alpha, batch_size=10, verbose=True,n_iter=100).fit(data_noisy)
components = sc.components_  #.reshape((K, h, w))

code = sc.transform(data_noisy)
patches = np.dot(code, components)
data_denoised =patches

print "final patch vectors: ",data_denoised.shape
#"""

#If you learn a model that producces an explicit image basis,
#load the basis elements into the array B below. The shape of the
#array should be the number of components in the basis times
#patch_size[0]*patch_size[1]*3 (the length of one row of data_noisy).
B = components

#This function takes your de-denoised data and re-assembles it into
#a complete color image
img_denoised  = util.patchToImage(util.vectorsToPatches(data_denoised,patch_size),(900,1200))

#These functions compute the MAE between the clean image,the noisy image and the de-noised image
print "Error of Noisy to Clean: %.4f"%util.eval_recon(img_clean,img_noisy)
print "Error of De-Noised to Clean: %.4f"%util.eval_recon(img_clean,img_denoised)

#Plot the clean and noisy images
plt.figure(0,figsize=(7,3))
util.plot_pair(img_clean,img_noisy,"Clean","Noisy")

#Plot the clean and de-noised images
plt.figure(1,figsize=(7,3))
util.plot_pair(img_clean,img_denoised,"Clean","De-Noised")

#Plot the image basis
plt.figure(2,figsize=(4,3))
util.plot_img_array(B,patch_size)
plt.suptitle("Image Basis")
plt.show()

