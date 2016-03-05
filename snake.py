from PIL import Image, ImageDraw
import numpy as np
from numpy.matlib import *
from numpy.linalg import *

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.pyplot import imshow
# %matplotlib inline

def load_image( infilename ) :
    img = Image.open( infilename ).convert('L')
    data = np.asarray( img, dtype="int32" )
    return data

def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )
    
# initial points
cx = 600
cy = 600
r = 500
radius = np.arange(0, 2*np.pi, 0.1);
x = cx+r*np.cos(radius)
y = cy+r*np.sin(radius)
coords = np.array([x,y], np.int32)
N = np.size(coords,1)

# load image
img = load_image('tree.jpg')

# draw image
fig, ax = plt.subplots()
ax.imshow(img, cmap = cm.Greys_r)
ax.plot(coords[0], coords[1], 'b')

# parameter
alpha = 0.05
beta = 0.0001
gamma =1
iterations = 1000

a = gamma*(2*alpha+6*beta)+1
b = gamma*(-alpha-4*beta)
c = gamma*beta

P = np.zeros(N)
P.fill(a)
P = np.diag(P)
print P

P1 = np.zeros(N)
P1.fill(b)
P1 = np.diag(P1)
print P1

P2 = np.zeros(N)
P2.fill(c)
P2 = np.diag(P2)
print P2

P1 = np.roll(P1,1,axis=0)
P = np.add(P, P1)
P1 = np.roll(P1,-2,axis=0)
P = np.add(P, P1)
P2 = np.roll(P2,2,axis=0)
P = np.add(P, P2)
P2 = np.roll(P2,-4,axis=0)
P = np.add(P, P2)

print P
P = inv(P)

(dx, dy) = np.gradient(img)


for ii in range(iterations):
    fex = dx[coords[0],coords[1]]
    fey = dy[coords[0],coords[1]]
    # Move control points
    coords[0] = dot(P,(coords[0]+gamma*fex))
    coords[1] = dot(P,(coords[1]+gamma*fey))
    if ii%10 == 0:
        ax.plot(coords[0], coords[1], 'b')
plt.show()