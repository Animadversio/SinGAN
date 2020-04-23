import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
from skimage.transform import rescale,resize,pyramid_gaussian,pyramid_reduce
import matplotlib.pylab as plt
import umap
#%%
from os.path import join
#%%
reducer = umap.UMAP(metric="correlation")
embedding = reducer.fit_transform(iris.data)
embedding.shape
#%%
face_attr_df = pd.read_csv(r"E:\Datasets\celeba-dataset\list_attr_celeba.csv")
#%%
celeba_dir = r"E:\Datasets\celeba-dataset\img_align_celeba\img_align_celeba"
imgs = []
for imgi in range(1,10001):
    img = imread(join(celeba_dir, r"%06d.jpg" % imgi))
    #print(img.shape)
    imgs.append(img)
#%%
# img_rd = rescale(img, 0.11, multichannel=True, anti_aliasing=True)
# print(img_rd.shape)
img_rs = resize(img, (24, 20, 3), anti_aliasing=True)
print(img_rs.shape)
plt.imshow(img_rs)
plt.show()
#%%
img_vec = []
for imgi in range(10000):
    # img = imread(join(celeba_dir, r"%06d.jpg" % (imgi+1)))
    # img_rs = resize(img, (24, 20, 3), anti_aliasing=True)
    img_rs = resize(imgs[imgi], (24, 20, 3), anti_aliasing=True)
    img_vec.append(img_rs.flatten())
img_vec = np.array(img_vec)
#%%
import time
t0 = time.time()
reducer = umap.UMAP(n_components=3, metric="correlation")
face_embed = reducer.fit_transform(X=img_vec)
print(time.time()-t0, "s")
#%%
plt.scatter(face_embed[:,0], face_embed[:,1],face_embed[:,1],c=face_attr_df.Male[:10000].to_numpy(),alpha=0.2)
plt.show()
#%%
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(face_embed[:,0], face_embed[:,1],face_embed[:,2],c=face_attr_df.Male[:10000].to_numpy(),alpha=0.2)