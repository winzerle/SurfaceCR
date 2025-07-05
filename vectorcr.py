# This file is part of SurfaceCR
# Copyright (c) 2025 Thomas Richter (thomas.richter@ovgu.de)
# Licensed under the MIT License – see LICENSE file for details.

import numpy as np
import pyvista as pv
from collections import defaultdict
from scipy.sparse import coo_matrix

'''

Main class defining the CR surface element for vector valued PDE's

The degrees of freedom store the co-normal components in each edge midpoint. 

Given an triangle (face) T, n_T is the oriented normal vector facing outside 
of the surface. T has three points (x_1,x_2,x_3) counter-clockwise oriented
and three edges (e_1=(x_1,x_2), e_2=(x_2,x_3), e_3=(x_3,x_1)). On each edge
- t_i = e_i is the tangential vector (in the plane)
- n_i = t_i x n_T is the normal vector (in the plane) facing outward. 
- c_i in {0,1} stores the edge orientation, i.e. for an edge e = T_1\cap T_2 
  which belongs to two elements, it once has the value 0, once 1. 

Given N edges, the discrete solution on triangle T is given as

u_T(x) = \sum_{i=0}^{N-1} phi_i(x) * c_i (u_i^n n_i + u_i^t t_i)

where phi_i(x) is the usual (scalar) Croizeix-Raviart basis function.

'''

class VectorSurfaceMeshCR:
    def __init__(self, mesh: pv.PolyData):
        self.mesh = mesh.triangulate()
        self.nodes = self.mesh.points
        self.faces = self.mesh.faces.reshape(-1, 4)[:, 1:]  # (n_faces, 3)
        self.n_faces = self.faces.shape[0]

        # Build edge-to-DoF map and store edge midpoints
        self.edge_to_index, self.edges, self.dof_coords = self._build_edges()
        self.n_edges = len(self.edges)
        self.n_dofs = 2*self.n_edges     # 2 dofs per edge

        # Compute mapping from each triangle to its local edge DoFs
        self.local_edge_map = self._build_local_dof_map()

        # Geometric quantities
        self.areas = self._compute_element_areas()
        self.edgelengths = self._compute_edge_lengths()
        self.normals = self._compute_element_normals()
        self.edge_tangents, self.edge_normals, self.edge_orientations, self.continuous_edge_normals, self.continuous_edge_tangents = self._compute_edge_tangents_and_orientations()



    def _build_edges(self):
        edge_to_index = {}
        edges = []
        dof_coords = []

        for f in self.faces:
            for a, b in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
                edge = tuple(sorted((a, b)))
                if edge not in edge_to_index:
                    edge_to_index[edge] = len(edges)
                    edges.append(edge)
                    midpoint = 0.5 * (self.nodes[edge[0]] + self.nodes[edge[1]])
#                    midpoint /= np.linalg.norm(midpoint) ### Nur auf Sphere!!!!
                    dof_coords.append(midpoint)

        return edge_to_index, np.array(edges), np.array(dof_coords)

    def _build_local_dof_map(self):
        """Map from face index to 3 global DoFs (edge midpoints)."""
        local_map = np.zeros((self.n_faces, 3), dtype=int)
        for i, f in enumerate(self.faces):
            for j, (a, b) in enumerate([(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]):
                edge = tuple(sorted((a, b)))
                local_map[i, j] = self.edge_to_index[edge]
        return local_map

    def _compute_element_areas(self):
        v0 = self.nodes[self.faces[:, 0]]
        v1 = self.nodes[self.faces[:, 1]]
        v2 = self.nodes[self.faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        return 0.5 * np.linalg.norm(cross, axis=1)

    def _compute_edge_lengths(self):
        return np.linalg.norm(self.nodes[self.edges[:,0]]-self.nodes[self.edges[:,1]],axis=1)


    def _compute_element_normals(self):   # normale auf dem element mit Kreuz-Produkt
        v0 = self.nodes[self.faces[:, 0]]
        v1 = self.nodes[self.faces[:, 1]]
        v2 = self.nodes[self.faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)  ### Orientierung???
        return normals / np.linalg.norm(normals, axis=1, keepdims=True)

    def _compute_edge_tangents_and_orientations(self):
        tangents = np.zeros((self.n_faces, 3, 3))
        enormals = np.zeros((self.n_faces, 3, 3))
        orientations = np.zeros((self.n_faces, 3), dtype=int)

        cen = np.zeros( (self.n_edges,3) )
        cet = np.zeros( (self.n_edges,3) )

        for f_idx, face in enumerate(self.faces):
            for j, (a, b) in enumerate([(face[0], face[1]),
                                        (face[1], face[2]),
                                        (face[2], face[0])]):
                edge = tuple(sorted((a, b)))
                
                tangent = self.nodes[b] - self.nodes[a]
                tangent /= np.linalg.norm(tangent)
                tangents[f_idx,j] = tangent

                enormal = np.cross(tangent, self.normals[f_idx])
                enormal /= np.linalg.norm(enormal)
                enormals[f_idx,j] = enormal

                orientations[f_idx, j] = 1 if (a, b) == edge else -1

                cen[self.edge_to_index[edge]] += 0.5 * enormal * orientations[f_idx,j]
                cet[self.edge_to_index[edge]] += 0.5 * tangent * orientations[f_idx,j]



        for i in range(self.n_edges):
            assert np.linalg.norm(cen[i])>0
            cen[i] /= np.linalg.norm(cen[i])
            assert np.linalg.norm(cet[i])>0
            cet[i] /= np.linalg.norm(cet[i])
            
        return tangents, enormals, orientations, cen, cet

    def summary(self):
        print(f"Triangles: {self.n_faces}")
        print(f"Edges (DoFs): {self.n_edges}")
        print(f"Element area: min={self.areas.min():.3e}, mean={self.areas.mean():.3e}")


    # ---- finite element stuff
    def interpolate_function(self, f):  
        """
        INterpolates the scalar functions f1(x, y, z) and f2 to the components of the finite element space.
   
        Parameters:
            f : callable, f(x, y, z) → float or np.ndarray of shape (n_points,)

        Returns:
            u_h : np.ndarray of shape (n_edges,), the interpolated DoF vector
        """
        xyz = np.array([self.dof_coords[:, 0], self.dof_coords[:, 1], self.dof_coords[:, 2]])

        u_h = np.zeros(self.n_dofs)
 
        u_h[:self.n_edges] = (self.continuous_edge_normals*f(xyz).T).sum(axis=1)  # projection of 3d function to normals
        u_h[self.n_edges:] = (self.continuous_edge_tangents*f(xyz).T).sum(axis=1)  # ... to tangents

        return np.asarray(u_h, dtype=float)

    def assemble_mass_and_stiffness(self):
        """Assemble global CR mass and stiffness matrices using edge midpoint DoFs."""
        row_idx, col_idx = [], []
        rowM_idx, colM_idx = [], []
        mass_vals, stiff_vals = [], []

        for tri_idx in range(self.n_faces):
            dofs = self.local_edge_map[tri_idx]     # (3,) global DoFs for triangle
            dofs_c = self.dof_coords[dofs]

            v_ids = self.faces[tri_idx]             # vertex indices (3,)
            v = self.nodes[v_ids]                   # vertex coordinates (3, 3)
            area = self.areas[tri_idx]

            edgenormals  = self.edge_normals[tri_idx]  # outward normal vectors at edge. 
            edgetangents = self.edge_tangents[tri_idx]  # outward normal vectors at edge. 
            edgeorientations = self.edge_orientations[tri_idx]

            # product of orientes (o_i n_i) * (o_j n_j)
            Nij = (np.diag(edgeorientations)@edgenormals)@(np.diag(edgeorientations)@edgenormals).T     
            Tij = (np.diag(edgeorientations)@edgetangents)@(np.diag(edgeorientations)@edgetangents).T
            NTij = (np.diag(edgeorientations)@edgenormals)@(np.diag(edgeorientations)@edgetangents).T
            #TNij = (np.diag(edgeorientations)@edgetangents)@(np.diag(edgeorientations)@edgenormals).T  ## transposed of NTij

            # CR basis gradients (shape: 3×3)
            grads = self._cr_basis_gradients(v, edgenormals, self.edgelengths[dofs], area, dofs_c)

            # Local CR mass matrix (lumped version for simplicity)
            M_loc = (area / 3.0) * np.eye(3)

            # Local CR stiffness matrix
            A_NN = area * Nij * (grads @ grads.T)  # (3×3)
            A_TT = area * Tij * (grads @ grads.T)  # (3×3)
            A_NT = area * NTij.T * (grads @ grads.T)  # (3×3)   ### Unklar, wo transponiert werden muss !!!!
            A_TN = area * NTij * (grads @ grads.T)  # (3×3)
            
            
            # Assemble into global matrix
            for i in range(3):
                for j in range(3):
                    rowM_idx.append(dofs[i])         # 1rst comp
                    colM_idx.append(dofs[j])

                    row_idx.append(dofs[i])         # 1rst comp
                    col_idx.append(dofs[j])
                    mass_vals.append(M_loc[i, j])
                    stiff_vals.append(A_NN[i, j])

                    ## cross   ### wierum transponieren???
                    row_idx.append(self.n_edges + dofs[i])         # 2-1
                    col_idx.append(dofs[j])
                    stiff_vals.append(A_NT[i, j])

                    row_idx.append(dofs[i])         # 1-2
                    col_idx.append(self.n_edges + dofs[j])
                    stiff_vals.append(A_TN[i, j])

                    rowM_idx.append(self.n_edges + dofs[i])         # 2nd comp
                    colM_idx.append(self.n_edges + dofs[j])
                    row_idx.append(self.n_edges + dofs[i])         # 2nd comp
                    col_idx.append(self.n_edges + dofs[j])
                    mass_vals.append(M_loc[i, j])
                    stiff_vals.append(A_TT[i, j])

        n = self.n_dofs
        M = coo_matrix((mass_vals, (rowM_idx, colM_idx)), shape=(n, n)).tocsr()
        A = coo_matrix((stiff_vals, (row_idx, col_idx)), shape=(n, n)).tocsr()
        return M, A

    def _cr_basis_gradients(self, v, edgenormals, edgesizes, area, dc):
        """
        Compute gradients of CR basis functions over one triangle.
        Each gradient is orthogonal to an edge, constant over the triangle.
        """
        grads = np.zeros((3, 3))
        for i in range(3): # ###, (a, b) in enumerate([(1, 2), (2, 0), (0, 1)]):  # opposite edges
            grads[i] = edgenormals[i] * edgesizes[i] / area
        return grads
    
    ## compute the h1 error between discrete uh and exact gradient Gu
    def h1error(self, uh, Gu):
        h1 = 0.0 # storing the result
       
        for tri_idx in range(self.n_faces):
            dofs = self.local_edge_map[tri_idx]     # (3,) global DoFs for triangle
            dofs_c = self.dof_coords[dofs]

            # midpoint of face
            facemid = self.dof_coords[dofs].sum(axis=0)/3. 
            area = self.areas[tri_idx]
            v_ids = self.faces[tri_idx]             # vertex indices (3,)
            v = self.nodes[v_ids]                   # vertex coordinates (3, 3)

            exact = Gu(facemid)                     # Exact gradient in face midpoint


            ## gradients of basis functions
            edgenormals  = self.edge_normals[tri_idx]  # outward normal vectors at edge. 
            edgetangents = self.edge_tangents[tri_idx]  # outward normal vectors at edge. 
            edgeorientations = self.edge_orientations[tri_idx]
            grads = self._cr_basis_gradients(v, edgenormals, self.edgelengths[dofs], area, dofs_c)

            ## grad of solution
            # 
            # u = (ut_i t_i + un_i n_i) c_i phi_i(x)    
            #
            # nabla u = (ut_i t_i + un_i n_i) c_i n_i^T phi'_i(x)
            Guh = np.zeros((3,3)) #uh[dofs]*edgeorientations * grads * 
            for i in range(3):
                Guh += uh[dofs[i]]*edgeorientations[i] * np.outer(edgenormals[i],grads[i])
                Guh += uh[dofs[i]+self.n_edges]*edgeorientations[i] * np.outer(edgetangents[i],grads[i])

            h1 += (((Guh-exact)*(Guh-exact)).sum()*area)
        return np.sqrt(h1)

    ### - vtk
    def write_cr_vertex_reconstruction(self, u_h, filename="cr_reconstructed.vtk"):
        """
        Reconstruct CR function at vertices using per-element interpolation,
        then average contributions from all adjacent triangles.
        """
        if len(u_h) != self.n_dofs:
            raise ValueError("Length of u_h does not match number of Vector-CR DoFs (edges)")

        vertex_sum1 = np.zeros(self.nodes.shape[0])
        vertex_sum2 = np.zeros(self.nodes.shape[0])
        vertex_sum3 = np.zeros(self.nodes.shape[0])
        
        vertex_count = np.zeros(self.nodes.shape[0], dtype=int)

        for tri_idx in range(self.n_faces):
            v_ids = self.faces[tri_idx]               # indices of triangle vertices
            v = self.nodes[v_ids]                     # shape: (3, 3)
            edge_ids = self.local_edge_map[tri_idx]   # CR DoFs

            ## normal and tangental components, properly oriented
            normal_comps  = u_h[edge_ids]# * MIT NORMALEN MULTIPLIZIEREN UND MIT TANGENTEN...                 # CR values at edge midpoints comp 1
            tangent_comps = u_h[edge_ids + self.n_edges]  # CR values at edge midpoints comp 2
#            normal_comps  = self.edge_orientations[tri_idx] * u_h[edge_ids]# * MIT NORMALEN MULTIPLIZIEREN UND MIT TANGENTEN...                 # CR values at edge midpoints comp 1
#            tangent_comps = self.edge_orientations[tri_idx] * u_h[edge_ids + self.n_edges]  # CR values at edge midpoints comp 2


            ## 3d vector field
            xyz = (self.continuous_edge_normals[edge_ids].T*normal_comps).T + (self.continuous_edge_tangents[edge_ids].T*tangent_comps).T

            # Get basis function coefficients: u = sum_i mid_vals[i] * phi_i(x)

            # Evaluate basis functions at triangle vertices
            # Compute barycentric coordinates of edge midpoints
            mids = 0.5 * (v[np.array([0, 1, 2])] + v[np.array([1, 2, 0])])  # midpoints

            # Construct matrix: phi_j(vertex_i)
            A = np.ones((3, 3))
            for j in range(3):
                A[j] = mids[j] - v[(j+2)%3]  # phi_j vanishes on vertex opposite to edge j

            # Solve for local linear function u(x) = a + b·x
            # Build system to fit mid_vals = a + b·x_mid
            X = np.hstack((np.ones((3, 1)), mids))  # [1, x, y, z] for each midpoint
            beta1, *_ = np.linalg.lstsq(X, xyz[:,0], rcond=None)  # beta = [a, b_x, b_y, b_z]
            beta2, *_ = np.linalg.lstsq(X, xyz[:,1], rcond=None)  # beta = [a, b_x, b_y, b_z]
            beta3, *_ = np.linalg.lstsq(X, xyz[:,2], rcond=None)  # beta = [a, b_x, b_y, b_z]

            # Evaluate u at each triangle vertex
            for local_idx, global_idx in enumerate(v_ids):
                x = self.nodes[global_idx]
                u_val1 = beta1[0] + beta1[1:].dot(x)
                u_val2 = beta2[0] + beta2[1:].dot(x)
                u_val3 = beta3[0] + beta3[1:].dot(x)
                vertex_sum1[global_idx] += u_val1
                vertex_sum2[global_idx] += u_val2
                vertex_sum3[global_idx] += u_val3
                vertex_count[global_idx] += 1

        # Average contributions from all adjacent triangles
        vertex_count[vertex_count == 0] = 1
        vertex_values1 = vertex_sum1 / vertex_count
        vertex_values2 = vertex_sum2 / vertex_count
        vertex_values3 = vertex_sum3 / vertex_count

        # Create PyVista mesh with vertex field
        cells = np.hstack([[3, *face] for face in self.faces])
        mesh = pv.PolyData(self.nodes, cells)

        vec = np.column_stack((vertex_values1, vertex_values2, vertex_values3))  # shape (n_points, 3)
        mesh.point_data["u"] = vec
        mesh.point_data.active_vectors_name = "u"  # tell VTK this is the default/active vector

        mesh.point_data["u1"] = vertex_values1
        mesh.point_data["u2"] = vertex_values2
        mesh.point_data["u3"] = vertex_values3
        mesh.save(filename)
#        print(f"Reconstructed CR function written to {filename}")
    
    
        # ---- Accessor Methods ---

    def get_dof_coordinates(self):
        return self.dof_coords  # shape: (n_edges, 3)

    def get_local_dof_map(self):
        return self.local_edge_map  # shape: (n_faces, 3)

    def get_element_normals(self):
        return self.normals  # shape: (n_faces, 3)

    def get_edge_tangents(self):
        return self.edge_tangents  # shape: (n_faces, 3, 3)

    def get_edge_orientations(self):
        return self.edge_orientations  # shape: (n_faces, 3)

    def get_elements(self):
        return self.faces

    def get_edges(self):
        return self.edges  # edge list as pairs of node indices

    def get_dof_count(self):
        return self.n_edges
