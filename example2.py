import torch
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import pylab as plt
from skimage.io import imread
from skimage.transform import resize

# READ IMAGES
cat = resize(imread("cat.jpeg"), (64,64,3))
mask = np.zeros(cat.shape[:2])
mask[30:35, 30:35] = 1

mask[35:40, 18:22] = 1

plt.subplot(121)
plt.imshow(mask)

plt.subplot(122)
plt.imshow(cat)

plt.show()

# Create Matrices
X = cat[np.newaxis]
X = np.transpose(X, (0, 3, 1, 2))
X = Variable(torch.FloatTensor(X))

Y = mask[np.newaxis, np.newaxis]
Y = Variable(torch.FloatTensor(Y))

# Create 3 parameter matrices
cn1 = Variable(torch.randn(5, 3, 3, 3), requires_grad=True)
cn2 = Variable(torch.randn(5, 5, 3, 3), requires_grad=True)
cn3 = Variable(torch.randn(1, 5, 3, 3), requires_grad=True)
    
opt = torch.optim.Adadelta([cn1,cn2, cn3], lr=1.0)

plt.ion()
fig = plt.figure()
for i in range(10000):
    opt.zero_grad()
    # Apply Convolution
    A = F.conv2d(X, cn1, padding=1)
    A = F.conv2d(A, cn2, padding=1)
    A = F.conv2d(A, cn3, padding=1) 

    pred = F.sigmoid(A)
    
    loss = (0.5*(A - Y)**2).sum()
    loss.backward()

    if (i%50) == 0:
        fig.clf()
        plt.suptitle("%d - loss %.3f" %  (i, loss.data[0]))
        plt.subplot(131)
        plt.imshow(cat)
        plt.subplot(132)
        plt.imshow(pred.data.numpy()[0,0])
        plt.subplot(133)
        plt.imshow(Y.data.numpy()[0,0])
        plt.show()
        plt.pause(0.0005)
        print ("%d - loss %.3f" %  (i, loss.data[0]))

    opt.step()



import pdb; pdb.set_trace()  # breakpoint ccf31f0a //
