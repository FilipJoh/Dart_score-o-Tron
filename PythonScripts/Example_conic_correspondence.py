import matplotlib.pyplot as plt
from copy import deepcopy

import numpy as np
import cv2

class Conic(object):
    def __init__(self, C, Z):
        # Ax**2 + Bxy + Cy**2 + Dx + Ey + F = 0.
        self.C = C
        self.Z = Z
    
    def sample(self, npoints=10):
        # Given: p * M * p.T = 0
        pass
    
    def planar_transform(self, R1, t1, R2, t2):
        n = np.expand_dims(np.asarray(((0),
                                       (0),
                                       (1))), axis=1)
        
        H_12 = R1.dot(np.eye(3) - ((t2  - t1).dot(n.T))/self.Z).dot(R2.T)
        new_C = np.linalg.inv(H_12).T.dot(self.C).dot(np.linalg.inv(H_12))
        
        discriminant = 2*new_C[0, 1]**2 - 4*new_C[0, 0]*new_C[1, 1]
        if np.linalg.det(new_C) == 0:
            ## Degenerate
            new_conic = None
        elif discriminant < 0:
            # Ellipse
            new_conic = Ellipse()
            new_conic.C = new_C
            new_conic.Z = self.Z
            new_conic.implicit_coefficients = {'A': new_C[0,0],
                                             'B': new_C[0,1]*2,
                                             'C': new_C[1,1],
                                             'D': new_C[0,2]*2,
                                             'E': new_C[1,2]*2,
                                             'F': new_C[2,2]}
            new_conic.set_parametrization()
    
        elif discriminant == 0:
            # Parabola
            new_conic = None
        elif discriminant > 0:
            # Hyperbola
            new_conic = None
        return new_conic
    
    def __get_parametrization(self):
        pass
    
    def set_parametrization(self):
        pass
    
    def plot(self):
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        for i in range(x.size):
            for j in range(y.size):
                vec = np.asarray([x[i], y[j], 1])
                val = (vec.T).dot(self.C).dot(vec)
                if val < 0:
                    plt.scatter(x[i], y[j], color='black', s=5)
        plt.grid('on')
        plt.axis([-10,10,-10,10])
        plt.axhline(y=0, color='k')
        plt.axvline(x=0, color='k')
        plt.legend()
        
        
        
        
        
        

class Ellipse(Conic):
    def __init__(self, C_x=0, C_y=0, R_x=1, R_y=1, theta=0, Z=1):
        self.C_x = C_x
        self.C_y = C_y
        self.R_x = R_x
        self.R_y = R_y
        self.theta = theta
        
        self.get_parametrization()
        matrix_representation = np.asarray(((self.implicit_coefficients['A'], self.implicit_coefficients['B']/2, self.implicit_coefficients['D']/2),
                                             (self.implicit_coefficients['B']/2, self.implicit_coefficients['C'], self.implicit_coefficients['E']/2),
                                             (self.implicit_coefficients['D']/2, self.implicit_coefficients['E']/2, self.implicit_coefficients['F'])))
        
        super().__init__(matrix_representation, Z)
        
    def get_parametrization(self):
        A = (self.R_y**2) * (np.cos(self.theta)**2) + (self.R_x**2) * (np.sin(self.theta))**2
        B = 2 * np.sin(self.theta) * np.cos(self.theta) * (self.R_y**2 - self.R_x**2)
        C = (self.R_y**2) * (np.sin(self.theta))**2 + (self.R_x**2) * (np.cos(self.theta))**2
        D = -(2*A*self.C_x + B*self.C_y)
        E = -(2*C*self.C_y + B*self.C_x)
        F = A*self.C_x**2 + B*self.C_x*self.C_y + C*self.C_y**2 - self.R_x**2*self.R_y**2
        
        self.implicit_coefficients =  {'A': A,
                                       'B': B,
                                       'C': C,
                                       'D': D,
                                       'E': E,
                                       'F': F}
    
    def set_parametrization(self):
        eq_2by2 = np.asarray(((self.implicit_coefficients['A'], self.implicit_coefficients['B']/2),
                               (self.implicit_coefficients['B']/2, self.implicit_coefficients['C'])))
        X_0 = np.linalg.solve(eq_2by2, np.asarray((-self.implicit_coefficients['D']/2, -self.implicit_coefficients['E']/2)))
        self.C_x = X_0[0]
        self.C_y = X_0[1]
        
        e, v = np.linalg.eig(eq_2by2)
        self.R_x = np.sqrt(e[1])
        self.R_y = np.sqrt(e[0])
        self.theta = np.arctan(v[1,0] / v[0,0])
        if self.theta < 0:
            self.theta +=  2*np.pi
        pass
    
    def plot(self, title):
        p = self.sample(1000)
        plt.figure()
        plt.scatter(p[0,:], p[1,:], label='parametrical')
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.axis([-10, 10, -10, 10])
        plt.grid('on')
        plt.title(title)

    def sample(self, npoints=10):
        # Generate uniformely distributed angles
        angles = 2*np.pi*np.random.rand(npoints)
        
        points = np.zeros((2, npoints))
        for i, angle in enumerate(angles):
            # get coordinates for unrotated, origin-centered ellipse
            x = self.R_x*np.cos(angle)
            y = self.R_y*np.sin(angle)
            
            # rotate coordinates
            x_rot = x*np.cos(self.theta) - y*np.sin(self.theta)
            y_rot = x*np.sin(self.theta) + y*np.cos(self.theta)
            
            # offset coordinates
            x_translated = x_rot + self.C_x
            y_translated = y_rot + self.C_y
            
            points[0, i] = x_translated
            points[1, i] = y_translated
        
        return points


e_1 = Ellipse(0, 5, 2, 2, 0.0*np.pi)
e_1.plot('Before homography')
        

R1 = np.asarray(((1, 0, 0),
                 (0, 1, 0),
                 (0, 0, 1)))

t1 = np.expand_dims(np.asarray(((0),
                                (0),
                                (0))), axis=1)

##No rotation
#R2 = np.asarray(((1, 0, 0),
#                 (0, 1, 0),
#                 (0, 0, 1)))

#Rotation about z-axis
theta_rot = np.pi/2
R2 = np.asarray(((np.cos(theta_rot), -np.sin(theta_rot), 0),
                 (np.sin(theta_rot), np.cos(theta_rot), 0),
                 (0, 0, 1)))

##Rotation about y-axis
#phi_rot = np.pi/16
#R2 = np.asarray(((np.cos(phi_rot), 0, np.sin(phi_rot)),
#                 (0, 1, 0),
#                 (-np.sin(phi_rot), 0, np.cos(phi_rot))))

t2 = np.expand_dims(np.asarray(((0),
                                (0),
                                (0))), axis=1)


e_2 = e_1.planar_transform(R1, t1, R2, t2)
e_2.plot('After homography')

