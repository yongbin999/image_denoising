import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.ndimage import imread
from sklearn import datasets

 
def imageToPatch(img,patch_size):
  #Convert an image of size NxM to an array of patches. H=patch_size[0] is the height of the patch.
  #W=patch_size[1] is the width of the patch. The number of patches is P = N*M/(H*W). H must divide N
  #and W must divide M. The shape of the returned array is (P,H,W,3). The inverese of this function is
  #pathToImage  
  NR,NC = img.shape[:2]
  if(not(0==NR % patch_size[0])):
    print("Error: Patch height %d must divide number of image rows %d"%(patch_size[0], NR))

  if(not(0==NC % patch_size[1])):
    print("Error: Patch width must divide number of image columns"%(patch_size[1], NC))

  NP = NR/patch_size[0] * NC/patch_size[1]

  data = np.zeros((NP,patch_size[0],patch_size[1],3))
  p=0
  for i in range(NR/patch_size[0]):
    for j in range(NC/patch_size[1]):
      data[p,:,:,:]= img[(i*patch_size[0]):((i+1)*patch_size[0]),(j*patch_size[1]):((j+1)*patch_size[1]),:]
      p=p+1
  return data

def patchToImage(patches,img_size):
  #Convert an array of patches to an image. Let the patches array have shape (P,H,W,3), N = img_sizep[0]
  #and M = img_size[1]. This function assembles the patches into an image in a way that inverts the  
  #imageToPatch function. H must divide N, W must divide M, and P must be N*M/(H*W).
  #The shape of the returned array is (N,M,3)
  patch_size=[0,0]
  NP,patch_size[0],patch_size[1] = patches.shape[:3]
  NR,NC = img_size
  if(not(0==NR % patch_size[0])):
    print("Error: Patch height %d must divide number of image rows %d"%(patch_size[0], NC))

  if(not(0==NC % patch_size[1])):
    print("Error: Patch width must divide number of image columns"%(patch_size[1], NC))

  img = np.zeros((NR,NC,3))
  p=0
  for i in range(NR/patch_size[0]):
    for j in range(NC/patch_size[1]):
      img[(i*patch_size[0]):((i+1)*patch_size[0]),(j*patch_size[1]):((j+1)*patch_size[1]),:]=patches[p,:,:,:]
      p=p+1
  return img

def patchesToVectors(patches):
  #Converts an array of patches of shape (P,H,W,3) to an array of shape (P,H*W*3) with the data for each
  #patch converted to a single long vector.
  N = patches.shape[0]  
  return patches.reshape(N,patches.shape[1]*patches.shape[2]*3)

def vectorsToPatches(vectors,patch_size):
  #Converts an array of shape (P,Q) to an array of shape (P,H,W,3) where
  #H = patch_size[0], W=patch_size[1] and Q = H*W*3. This function is the inverse
  #of patchesToVectors

  N = vectors.shape[0]  
  return vectors.reshape((N,patch_size[0],patch_size[1],3))

def plot_pair(A,B,TitleA,TitleB):
  #This function displays two images A and B side-by-side   
  plt.subplot(1,2,1)
  plt.imshow(A,interpolation="nearest")
  plt.xticks(())
  plt.yticks(())
  plt.title(TitleA)

  plt.subplot(1,2,2)
  plt.imshow(B,interpolation="nearest")
  plt.xticks(())
  plt.yticks(())
  plt.title(TitleB)
  
  plt.tight_layout()

def plot_img_array(B,patch_size,grey=False):
  #This function displays the first 100 elements of an image patch bases. 
  #B is expected to have shape (N,Q) where Q = H*W*3, where H = patch_size[0], and
  #W=patch_size[1]. Each row of B is converted to a (HxWx3) array, scaled to be positive,
  #and then displayed as an image. 
  S = min(10,np.ceil(np.sqrt(B.shape[0]))) 
  N = min(100,B.shape[0])
  for i, comp in enumerate(B[:N]):
    plt.subplot(S, S, i + 1)
    comp=comp-min(comp.flatten())
    comp=comp/max(comp.flatten())
    if(grey==False):
      plt.imshow(comp.reshape((patch_size[0],patch_size[1],3)),interpolation="nearest")
    else:
      plt.imshow(comp.reshape((patch_size[0],patch_size[1])),interpolation="nearest",cmap='gray')
      
    plt.xticks(())
    plt.yticks(())

def eval_recon(img_est, img_clean):
    #This function evaluates the MAE econstruction error given a 
    #clean image and an estimated image as inputs 
    return np.mean(np.abs(img_est.flatten()-img_clean.flatten()))


def loadDataQ1():
  #This function load the images dataset for Q1
  imgs = datasets.load_digits().images
  imgs = imgs / np.max(imgs.flatten())
  imgs_vectors = imgs.reshape((len(imgs),-1))
  return imgs, imgs_vectors


def loadDataQ2(patch_size):
  #This function loads the example image for Q2, adds noise,
  #and converts it to the required representations for
  #denoising
  np.random.seed(0)
  imageName  = "../../Data/StarryNight.jpg"
  img        = imread(imageName,flatten=False)/255.0
  img_noisy  = img + 0.2 * np.random.randn(*img.shape)
  img_noisy[img_noisy>1] = 1
  img_noisy[img_noisy<0] = 0
  data       = patchesToVectors(imageToPatch(img,patch_size))
  data_noisy = patchesToVectors(imageToPatch(img_noisy,patch_size))

  return data_noisy, data, img_noisy, img
