# This file is part of SurfaceCR
# Copyright (c) 2025 Thomas Richter (thomas.richter@ovgu.de)
# Licensed under the MIT License – see LICENSE file for details.

'''

Main script running a convergence study and plotting the results. 

'''


import pyvista as pv
import numpy as np
import vectorcr as vcr
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

### exact solution. Will be projected to the tangential plane whenever used
def u(xyz):
    x = xyz[0,:]
    y = xyz[1,:]
    z = xyz[2,:]
    return np.array([y*z,-x*z,z*z])


## Right hand side: f = - Delta u + 1/10 * u
def f(xyz):
    x = xyz[0,:]
    y = xyz[1,:]
    z = xyz[2,:]

    # see symbolic program torus.py for computing a rhs
    return np.array([z*(0.4*x*z**2*(1.0*x**4 + 2.0*x**2*y**2 + 1.0*y**4) + np.sqrt(x**2 + y**2)*(-0.4*x*z**2 + 0.1*y)*(1.0*x**4 + 2.0*x**2*y**2 + 1.0*y**4) + np.sqrt(x**2 + y**2)*(240.0*x**7 + 720.0*x**5*y**2 - 768.0*x**5*np.sqrt(x**2 + y**2) + 820.0*x**5 + 20.0*x**4*y + 720.0*x**3*y**4 - 1536.0*x**3*y**2*np.sqrt(x**2 + y**2) + 1640.0*x**3*y**2 - 332.0*x**3*np.sqrt(x**2 + y**2) + 44.0*x**3 + 40.0*x**2*y**3 - 16.0*x**2*y*np.sqrt(x**2 + y**2) + 240.0*x*y**6 - 768.0*x*y**4*np.sqrt(x**2 + y**2) + 820.0*x*y**4 - 332.0*x*y**2*np.sqrt(x**2 + y**2) + 44.0*x*y**2 - 3.0*x*np.sqrt(x**2 + y**2) + 20.0*y**5 - 16.0*y**3*np.sqrt(x**2 + y**2)))/(np.sqrt(x**2 + y**2)*(1.0*x**4 + 2.0*x**2*y**2 + 1.0*y**4)),z*(0.4*y*z**2*(1.0*x**4 + 2.0*x**2*y**2 + 1.0*y**4) - (0.1*x + 0.4*y*z**2)*np.sqrt(x**2 + y**2)*(1.0*x**4 + 2.0*x**2*y**2 + 1.0*y**4) + np.sqrt(x**2 + y**2)*(240.0*x**6*y - 20.0*x**5 + 720.0*x**4*y**3 - 768.0*x**4*y*np.sqrt(x**2 + y**2) + 820.0*x**4*y - 40.0*x**3*y**2 + 16.0*x**3*np.sqrt(x**2 + y**2) + 720.0*x**2*y**5 - 1536.0*x**2*y**3*np.sqrt(x**2 + y**2) + 1640.0*x**2*y**3 - 332.0*x**2*y*np.sqrt(x**2 + y**2) + 44.0*x**2*y - 20.0*x*y**4 + 16.0*x*y**2*np.sqrt(x**2 + y**2) + 240.0*y**7 - 768.0*y**5*np.sqrt(x**2 + y**2) + 820.0*y**5 - 332.0*y**3*np.sqrt(x**2 + y**2) + 44.0*y**3 - 3.0*y*np.sqrt(x**2 + y**2)))/(np.sqrt(x**2 + y**2)*(1.0*x**4 + 2.0*x**2*y**2 + 1.0*y**4)),(1008.0*x**4 + 2016.0*x**2*y**2 + 1112.0*x**2 + 1008.0*y**4 + 1112.0*y**2 + np.sqrt(x**2 + y**2)*(-240.0*x**4 - 480.0*x**2*y**2 - 1572.0*x**2 - 240.0*y**4 - 1572.0*y**2 - 0.4*z**4 + 0.1*z**2 - 342.0) + 33.0)/np.sqrt(x**2 + y**2)])

### Gradient (here, xyz is a single point, no vector) See torus.py 
def Gu(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    return np.array([[z*(-48.0*x**8 - 144.0*x**6*y**2 + 192.0*x**6*np.sqrt(x**2 + y**2) - 272.0*x**6 - 4.0*x**5*y - 144.0*x**4*y**4 + 384.0*x**4*y**2*np.sqrt(x**2 + y**2) - 540.0*x**4*y**2 + 160.0*x**4*np.sqrt(x**2 + y**2) - 33.0*x**4 - 8.0*x**3*y**3 + 4.0*x**3*y*np.sqrt(x**2 + y**2) - 48.0*x**2*y**6 + 192.0*x**2*y**4*np.sqrt(x**2 + y**2) - 264.0*x**2*y**4 + 148.0*x**2*y**2*np.sqrt(x**2 + y**2) - 22.0*x**2*y**2 - 4.0*x*y**5 + 4.0*x*y**3*np.sqrt(x**2 + y**2) + 4.0*y**6 - 12.0*y**4*np.sqrt(x**2 + y**2) + 11.0*y**4 - 3.0*y**2*np.sqrt(x**2 + y**2))/(1.0*x**4 + 2.0*x**2*y**2 + 1.0*y**4),z*(-48.0*x**7*y - 4.0*x**6 - 144.0*x**5*y**3 + 192.0*x**5*y*np.sqrt(x**2 + y**2) - 276.0*x**5*y - 16.0*x**4*y**2 + 8.0*x**4*np.sqrt(x**2 + y**2) - 3.0*x**4 - 144.0*x**3*y**5 + 384.0*x**3*y**3*np.sqrt(x**2 + y**2) - 552.0*x**3*y**3 + 172.0*x**3*y*np.sqrt(x**2 + y**2) - 44.0*x**3*y - 20.0*x**2*y**4 + 20.0*x**2*y**2*np.sqrt(x**2 + y**2) - 6.0*x**2*y**2 - 48.0*x*y**7 + 192.0*x*y**5*np.sqrt(x**2 + y**2) - 276.0*x*y**5 + 172.0*x*y**3*np.sqrt(x**2 + y**2) - 44.0*x*y**3 + 3.0*x*y*np.sqrt(x**2 + y**2) - 8.0*y**6 + 12.0*y**4*np.sqrt(x**2 + y**2) - 3.0*y**4)/(1.0*x**4 + 2.0*x**2*y**2 + 1.0*y**4),48.0*x**5 - 240.0*x**5/np.sqrt(x**2 + y**2) + 96.0*x**3*y**2 - 480.0*x**3*y**2/np.sqrt(x**2 + y**2) + 464.0*x**3 - 432.0*x**3/np.sqrt(x**2 + y**2) + 8.0*x**2*y - 20.0*x**2*y/np.sqrt(x**2 + y**2) + 48.0*x*y**4 - 240.0*x*y**4/np.sqrt(x**2 + y**2) + 464.0*x*y**2 - 432.0*x*y**2/np.sqrt(x**2 + y**2) + 193.0*x - 33.0*x/np.sqrt(x**2 + y**2) + 8.0*y**3 - 20.0*y**3/np.sqrt(x**2 + y**2) + 15.0*y - 3.0*y/np.sqrt(x**2 + y**2)],[z*(-48.0*x**7*y + 8.0*x**6 - 144.0*x**5*y**3 + 192.0*x**5*y*np.sqrt(x**2 + y**2) - 276.0*x**5*y + 20.0*x**4*y**2 - 12.0*x**4*np.sqrt(x**2 + y**2) + 3.0*x**4 - 144.0*x**3*y**5 + 384.0*x**3*y**3*np.sqrt(x**2 + y**2) - 552.0*x**3*y**3 + 172.0*x**3*y*np.sqrt(x**2 + y**2) - 44.0*x**3*y + 16.0*x**2*y**4 - 20.0*x**2*y**2*np.sqrt(x**2 + y**2) + 6.0*x**2*y**2 - 48.0*x*y**7 + 192.0*x*y**5*np.sqrt(x**2 + y**2) - 276.0*x*y**5 + 172.0*x*y**3*np.sqrt(x**2 + y**2) - 44.0*x*y**3 + 3.0*x*y*np.sqrt(x**2 + y**2) + 4.0*y**6 - 8.0*y**4*np.sqrt(x**2 + y**2) + 3.0*y**4)/(1.0*x**4 + 2.0*x**2*y**2 + 1.0*y**4),z*(-48.0*x**6*y**2 + 4.0*x**6 + 4.0*x**5*y - 144.0*x**4*y**4 + 192.0*x**4*y**2*np.sqrt(x**2 + y**2) - 264.0*x**4*y**2 - 12.0*x**4*np.sqrt(x**2 + y**2) + 11.0*x**4 + 8.0*x**3*y**3 - 4.0*x**3*y*np.sqrt(x**2 + y**2) - 144.0*x**2*y**6 + 384.0*x**2*y**4*np.sqrt(x**2 + y**2) - 540.0*x**2*y**4 + 148.0*x**2*y**2*np.sqrt(x**2 + y**2) - 22.0*x**2*y**2 - 3.0*x**2*np.sqrt(x**2 + y**2) + 4.0*x*y**5 - 4.0*x*y**3*np.sqrt(x**2 + y**2) - 48.0*y**8 + 192.0*y**6*np.sqrt(x**2 + y**2) - 272.0*y**6 + 160.0*y**4*np.sqrt(x**2 + y**2) - 33.0*y**4)/(1.0*x**4 + 2.0*x**2*y**2 + 1.0*y**4),48.0*x**4*y - 240.0*x**4*y/np.sqrt(x**2 + y**2) - 8.0*x**3 + 20.0*x**3/np.sqrt(x**2 + y**2) + 96.0*x**2*y**3 - 480.0*x**2*y**3/np.sqrt(x**2 + y**2) + 464.0*x**2*y - 432.0*x**2*y/np.sqrt(x**2 + y**2) - 8.0*x*y**2 + 20.0*x*y**2/np.sqrt(x**2 + y**2) - 15.0*x + 3.0*x/np.sqrt(x**2 + y**2) + 48.0*y**5 - 240.0*y**5/np.sqrt(x**2 + y**2) + 464.0*y**3 - 432.0*y**3/np.sqrt(x**2 + y**2) + 193.0*y - 33.0*y/np.sqrt(x**2 + y**2)],[48.0*x**5 - 240.0*x**5/np.sqrt(x**2 + y**2) + 96.0*x**3*y**2 - 480.0*x**3*y**2/np.sqrt(x**2 + y**2) + 464.0*x**3 - 432.0*x**3/np.sqrt(x**2 + y**2) - 4.0*x**2*y + 12.0*x**2*y/np.sqrt(x**2 + y**2) + 48.0*x*y**4 - 240.0*x*y**4/np.sqrt(x**2 + y**2) + 464.0*x*y**2 - 432.0*x*y**2/np.sqrt(x**2 + y**2) + 193.0*x - 33.0*x/np.sqrt(x**2 + y**2) - 4.0*y**3 + 12.0*y**3/np.sqrt(x**2 + y**2) - 11.0*y + 3.0*y/np.sqrt(x**2 + y**2),48.0*x**4*y - 240.0*x**4*y/np.sqrt(x**2 + y**2) + 4.0*x**3 - 12.0*x**3/np.sqrt(x**2 + y**2) + 96.0*x**2*y**3 - 480.0*x**2*y**3/np.sqrt(x**2 + y**2) + 464.0*x**2*y - 432.0*x**2*y/np.sqrt(x**2 + y**2) + 4.0*x*y**2 - 12.0*x*y**2/np.sqrt(x**2 + y**2) + 11.0*x - 3.0*x/np.sqrt(x**2 + y**2) + 48.0*y**5 - 240.0*y**5/np.sqrt(x**2 + y**2) + 464.0*y**3 - 432.0*y**3/np.sqrt(x**2 + y**2) + 193.0*y - 33.0*y/np.sqrt(x**2 + y**2),-256*x**2*z**5 + 48*x**2*z**3 - 256*y**2*z**5 + 48*y**2*z**3 - 256*z**7 + 512*z**5*np.sqrt(x**2 + y**2) - 96*z**5 - 96*z**3*np.sqrt(x**2 + y**2) + 16*z**3 + 2*z]]) 


## just to store results for visualization
H   = [] # mesh sizes
El2 = [] # L2 errors
Eh1 = [] # h1 errors

for i,l in enumerate(range(2,7)):  # run on sequence of meshes
    surface = pv.ParametricTorus(u_res=2*2**l,v_res=2**l,w_res=2**l)  # generate the mesh
    fem = vcr.VectorSurfaceMeshCR(surface)                            # init CR element

    M, A = fem.assemble_mass_and_stiffness()                          # mass & stiffness matrix
    rhs = M @ fem.interpolate_function(f)                             # compute rhs
    uh = spsolve(A+ 0.1 * M,rhs)                                      # solve problem
    
    fem.write_cr_vertex_reconstruction(uh,'u_{}.vtk'.format(l))       # vtk output of solution

    eh = fem.interpolate_function(u) - uh                             # compute error
    fem.write_cr_vertex_reconstruction(eh,'e_{}.vtk'.format(l))       # output error as vtk

    H.append(np.max(fem.edgelengths))                                 # get mesh size
    El2.append(np.linalg.norm(eh,ord=np.inf))                         # compute L2 error
    Eh1.append(fem.h1error(uh,Gu))                                    # compute the H1 error
    print(H[i],El2[i],Eh1[i],sep='\t')


# plot errors
plt.loglog(H,El2,'-*',label='L2 error')
plt.loglog(H,Eh1,'-*',label='H1 error')
plt.loglog(H,np.array(H)**2 * El2[0]*0.8/H[0]/H[0],label='quadratic')
plt.loglog(H,np.array(H) * Eh1[0]*1.2/H[0],label='linear')
plt.legend()
plt.show()
