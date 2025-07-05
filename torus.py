# This file is part of SurfaceCR
# Copyright (c) 2025 Thomas Richter (thomas.richter@ovgu.de)
# Licensed under the MIT License – see LICENSE file for details.

'''

This script uses sympy to generate an analytical solution
and the corresponding right hand side. The output can be
copied to the main file surface.py 


Two examples describe simple solutions on the sphere and on the torus.

'''

from sympy import *

### Example 1
# Defines the surface as zero contour. 
# The gradient must give the outward facing unit normal vector field
def sphere(x,y,z):
    # Sphere with R = 1
    return 1/2 * (x*x+y*y+z*z-1)

def spheresimplify(x,y,z): # Rule for simplifying the final result on the surface
    return [z**2,1-x**2-y**2]
 
### Example 2
def torus(x,y,z):
    # Torus with R = 1 and r = 1/2. 
    # Gradient is given a unit outward facing normal vector field 
    return -(1/4 - z**2 - (sqrt(x**2+y**2)-1)**2)  # torus R=1 and r=1/2, outward normals with |n|=1

def torussimplify(x,y,z): # Rule for simplifying the final result on the surface
    return [z**2,1/4-(sqrt(x**2+y**2)-1)**2]


######
###### 

# returns the normal vector field
def normal(S, x,y,z):
    return Matrix([diff(S(x,y,z),x), diff(S(x,y,z),y), diff(S(x,y,z),z)])

# Gives the projection to the normal plane, I - N N.T for the surface defined by S(...)
def P(S, x,y,z):
    return simplify(eye(3)  - normal(S,x,y,z)*normal(S,x,y,z).T)

# Computes the Surface Gradient of a function. 
# The function must be in the tangential plane
# P is the proection to the normal plane
def SurfaceGradient(U, P, x,y,z):
    G = Matrix([[diff(U[0,0],x),diff(U[0,0],y),diff(U[0,0],z)],
                [diff(U[1,0],x),diff(U[1,0],y),diff(U[1,0],z)],
                [diff(U[2,0],x),diff(U[2,0],y),diff(U[2,0],z)]])
    return simplify(P*G*P)

def SurfaceLaplace(U, Ssimp, P, x, y, z):
    Gt = SurfaceGradient(U, P, x, y, z)
    div = []
    for k in [0,1,2]:
        GGt = SurfaceGradient(Gt.row(k).T, P, x, y, z)
        lap = simplify(GGt[0,0]+GGt[1,1]+GGt[2,2]).subs(Ssimp(x,y,z)[0],Ssimp(x,y,z)[1])
        div.append(simplify(lap))
    return Matrix(div)
## 
sx, sy, sz = symbols('x y z')


#S, Ssimp = sphere, spheresimplify
S, Ssimp = torus, torussimplify

P = P(S,sx,sy,sz)

U = Matrix([sy*sz,-sx*sz,sz*sz]) # Vector field

Ut = simplify(P*U)     # projected to surface

S = simplify(SurfaceGradient(Ut, P, sx, sy, sz).subs(Ssimp(sx,sy,sz)[0],Ssimp(sx,sy,sz)[1]))

Sxx = '{}'.format(S[0,0]).replace('sqrt','np.sqrt')
Sxy = '{}'.format(S[0,1]).replace('sqrt','np.sqrt')
Sxz = '{}'.format(S[0,2]).replace('sqrt','np.sqrt')
Syx = '{}'.format(S[1,0]).replace('sqrt','np.sqrt')
Syy = '{}'.format(S[1,1]).replace('sqrt','np.sqrt')
Syz = '{}'.format(S[1,2]).replace('sqrt','np.sqrt')
Szx = '{}'.format(S[2,0]).replace('sqrt','np.sqrt')
Szy = '{}'.format(S[2,1]).replace('sqrt','np.sqrt')
Szz = '{}'.format(S[2,2]).replace('sqrt','np.sqrt')


print('return np.array([[{},{},{}],[{},{},{}],[{},{},{}]])'.format(Sxx,Sxy,Sxz,Syx,Syy,Syz,Szx,Szy,Szz))

L = SurfaceLaplace(Ut, Ssimp, P, sx, sy, sz) # Surface Gradient of it

# -Delta u + 1/10 * u
rhs = simplify(-L + 1/10 * Ut)
dxx = '{}'.format(rhs[0,0]).replace('sqrt','np.sqrt')
dyy = '{}'.format(rhs[1,0]).replace('sqrt','np.sqrt')
dzz = '{}'.format(rhs[2,0]).replace('sqrt','np.sqrt')

print('return np.array([{},{},{}])'.format(dxx,dyy,dzz))
