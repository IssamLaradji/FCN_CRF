import torch
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import pylab as plt
from skimage.io import imread
from skimage.transform import resize

# READ IMAGES
cat = resize(imread("cat.jpeg"), (256,256,3))
dog = resize(imread("dog.jpeg"), (256,256,3))

plt.subplot(121)
plt.imshow(cat)

plt.subplot(122)
plt.imshow(dog)

plt.show()

# Create Matrices
X = np.vstack([cat[np.newaxis], dog[np.newaxis]])
X = np.transpose(X, (0, 3, 1, 2))
X = Variable(torch.FloatTensor(X))

y = Variable(torch.FloatTensor([0, 1]))

# Create 3 parameter matrices
cn1 = Variable(torch.randn(5, 3, 3, 3), requires_grad=True)
cn2 = Variable(torch.randn(5, 5, 3, 3), requires_grad=True)
cn3 = Variable(torch.randn(5, 5, 3, 3), requires_grad=True)
    
opt = torch.optim.Adadelta([cn1,cn2, cn3], lr=1.0)
import pdb; pdb.set_trace()  # breakpoint 548579e1 //

for i in range(10000):
    opt.zero_grad()
    # Apply Convolution
    A = F.conv2d(X, cn1)
    # A = F.conv2d(A, cn2)
    # A = F.conv2d(A, cn3) 

    pred = F.sigmoid(torch.max(A.view(2, -1), 1)[0])
    
    loss = (0.5*(pred - y)**2).sum()
    loss.backward()

    if (i%50) == 0:
        print ("%d - loss %.3f - True: %s - Pred: %s" % 
        (i, loss.data[0], y.data.numpy(), pred.data.numpy()))

    opt.step()



import pdb; pdb.set_trace()  # breakpoint ccf31f0a //
