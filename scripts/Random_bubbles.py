#######################################################################################
# Class for producing box containing randomly located disks/bubbles of a given radius #
#######################################################################################
# Handles periodic boundary conditions
#
# Written for 2D by Jonathan Pritchard (2017) - upgraded to 3D by Adélie Gorce (2018)
#

import numpy as np
import numpy.random as npr
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import sqrt, pi
import os
import sys, getopt
import time 
import pylab as pl
from mpl_toolkits.axes_grid1 import ImageGrid
from astropy.io import ascii
from astropy.table import Table, Column

class RandomBubbles:
    """
    Produce a random box filled with bubbles in 3D or disks in 2D
    """

    def __init__(self, DIM=512, nb=20, radius=20., NDIM = 2, nooverlap = False, periodic=True):
        """
        Initialise a DIM ^ NDIM box with random bubbles until given number of bubbles (nb) has been reached
    If 'periodic = True', the box had periodic boundary conditions
    If 'nooverlap = True', bubbles are not allowed to overlap (expect a long computing time if you have a large filling fraction)
        """
        
        self.NDIM = NDIM  #number of dimensions e.g. 2D or 3D
        self.DIM = DIM   #Number of cells per dimension
        self.nooverlap = nooverlap #Overlapping bubbles allowed or not 
        self.periodic = periodic #Periodic boundary conditions
        self.nb= int(nb) #Number of bubbles in the box
        self.radius=radius #Radius (in pixels) of all bubbles in the box

        print("%iD box with %i cells and %i bubbles of radius %i" %(self.NDIM, self.DIM, self.nb,self.radius))
        # print("periodicity and nooverlap are (%i,%i)" %(periodic, nooverlap))

        if self.NDIM == 2:
            self.box = np.zeros([DIM, DIM])
        elif self.NDIM ==3:
            self.box = np.zeros([DIM, DIM, DIM])
        else:
            raise Exception("NDIM must be 2 or 3" % (NDIM))

        self.bubbles = []

        #Add bubbles to get to target number
        self.increase_bubble_nb(self.nb)


    def summary(self):
        """
        Update summary statistics
        """

        self.box[self.box>1.]=1. #avoid pixel value to exceed 1 in overlap zones (pixel value <-> ionisation level)
        
        # print("Number of bubbles in=", len(self.bubbles))
        print("Box mean = ", self.box.mean())

        #Show the slice
        # plt.ion()
        # plt.ioff()
        colours=['midnightblue','lightyellow']
        bounds=[0,1]
        cmap = mpl.colors.ListedColormap(colours)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        self.xhII=np.round(self.box.mean(),decimals=3)


    def increase_bubble_nb(self, target):
        """
        Add randomly located and sized bubbles until reach
        desired number of bubbles
        """

        R=self.radius

        count=0

        while (count < target):

            #random position for the bubble
            x = npr.randint(self.DIM, size = self.NDIM)

            #Use mask to ID pixels in this ellipse
            bmask = self.bubble_mask(x, R)
            test=0
            if self.nooverlap:
        #check for overlaps
                if np.any(self.box[bmask.astype(np.bool)]):
                    continue

        #add bubble to whole box
            self.box=np.add(self.box,bmask).astype(np.bool)
            self.box=self.box.astype(int)

            #Store bubble locations
            self.bubbles.append(x)

            count=count+1
    
            if count % 100 == 0:
                print('For %i bubbles, xhII = %.2f \n' %(count, self.box.mean())) #keeps track of process

        self.summary()

    
    def bubble_mask(self, x, R):
        #wrapper to handle different dimensionality
        if (self.NDIM == 2):
            return self.disk_mask(x,R)
        elif(self.NDIM == 3):
            return self.sphere_mask(x,R)
        else:
            raise Exception ("NDIM is not 2 or 3")

    
    def disk_mask(self, pos, R):
    #generates mask corresponding to a 2D ionised disk
        #pos is coordinates of the centre of the bubble
        #R is its radius

        full_struct = np.zeros([self.DIM, self.DIM])

        #Creates a disk at centre of smaller structure to avoid generating another whole box: just enough to contain the disk
        structsize = int(2*R+6)
        x0 = y0 = int(structsize/2)
        struct = np.zeros((structsize, structsize))
        x, y = np.indices((structsize, structsize))
        mask = (x - structsize/2)**2 + (y - structsize/2)**2 <= R**2 #puts the disk in the middle of new box
        struct[mask] = 1
        
        #Now work out coordinate shift to move centre to pos    
        xmov = [pos[0] - x0, pos[0] + x0]
        ymov = [pos[1] - y0, pos[1] + y0]

        #if struct goes out of the box
        xmin=max(xmov[0],0)
        xmax=min(xmov[1],self.DIM)
        ymin=max(ymov[0],0)
        ymax=min(ymov[1],self.DIM)

    #periodic boundary conditions
        if self.periodic:
            if (xmov[0]<0):
                extra_struct=struct[0:abs(xmov[0]),abs(min(0,ymov[0])):min(structsize,self.DIM-ymov[0])]
                full_struct[self.DIM-abs(xmov[0]):self.DIM,ymin:ymax]=np.add(full_struct[self.DIM-abs(xmov[0]):self.DIM,ymin:ymax],extra_struct)
            if (xmov[1]>self.DIM):
                extra_struct=struct[structsize-(xmov[1]-self.DIM):structsize,abs(min(0,ymov[0])):min(structsize,structsize+self.DIM-ymov[1])]
                full_struct[0:xmov[1]-self.DIM,ymin:ymax]=np.add(full_struct[0:xmov[1]-self.DIM,ymin:ymax],extra_struct)
            if (ymov[0]<0):
                extra_struct=struct[abs(min(0,xmov[0])):min(structsize,self.DIM-xmov[0]),0:abs(ymov[0])]
                full_struct[xmin:xmax,self.DIM-abs(ymov[0]):self.DIM]=np.add(full_struct[xmin:xmax,self.DIM-abs(ymov[0]):self.DIM],extra_struct)
            if (ymov[1]>self.DIM):
                extra_struct=struct[abs(min(0,xmov[0])):min(structsize,structsize+self.DIM-xmov[1]),structsize-(ymov[1]-self.DIM):structsize]
                full_struct[xmin:xmax,0:ymov[1]-self.DIM]=np.add(full_struct[xmin:xmax,0:ymov[1]-self.DIM],extra_struct)

        #truncated struct if some part is outside the full struct
        small_struct=struct[abs(xmov[0]-xmin):structsize-abs(xmov[1]-xmax), abs(ymov[0]-ymin):structsize-abs(ymov[1]-ymax)]
        #add to previous box 
        full_struct[xmin:xmax,ymin:ymax] = np.add(full_struct[xmin:xmax,ymin:ymax],small_struct) 
        
        return full_struct


    def sphere_mask(self, pos, R):
    #generates mask corresponding to a 3D ionised sphere
        #pos is coordinates of the centre of the bubble
        #R is its radius

        full_struct = np.zeros([self.DIM, self.DIM, self.DIM])

        #Creates a disk at centre of smaller structure to avoid generating another whole box: just enough to contain the disk
        structsize = int(2*R+6)
        x0 = y0 = z0 = int(structsize/2)
        struct = np.zeros((structsize, structsize, structsize))
        x, y, z = np.indices((structsize, structsize, structsize))
        mask = (x - structsize/2)**2 + (y - structsize/2)**2 + (z - structsize/2)**2<= R**2
        struct[mask] = 1

        #Now work out coordinate shift to move centre to pos
        xmov = [pos[0] - x0,pos[0]+x0]
        ymov = [pos[1] - y0,pos[1]+y0]
        zmov = [pos[2] - z0,pos[2]+z0]

        #if struct goes out of the box
        xmin=max(xmov[0],0)
        xmax=min(xmov[1],self.DIM)
        ymin=max(ymov[0],0)
        ymax=min(ymov[1],self.DIM)
        zmin=max(zmov[0],0)
        zmax=min(zmov[1],self.DIM)

    #periodic boundary conditions
        if self.periodic:
            if (xmov[0]<0):
                extra_struct=struct[0:abs(xmov[0]),abs(min(0,ymov[0])):min(structsize,self.DIM-ymov[0]),abs(min(0,zmov[0])):min(structsize,self.DIM-zmov[0])]
                full_struct[self.DIM-abs(xmov[0]):self.DIM,ymin:ymax,zmin:zmax]=np.add(full_struct[self.DIM-abs(xmov[0]):self.DIM,ymin:ymax,zmin:zmax],extra_struct)
            if (xmov[1]>self.DIM):
                extra_struct=struct[structsize-(xmov[1]-self.DIM):structsize,abs(min(0,ymov[0])):min(structsize,structsize+self.DIM-ymov[1]),abs(min(0,zmov[0])):min(structsize,structsize+self.DIM-zmov[1])]
                full_struct[0:xmov[1]-self.DIM,ymin:ymax,zmin:zmax]=np.add(full_struct[0:xmov[1]-self.DIM,ymin:ymax,zmin:zmax],extra_struct)
            if (ymov[0]<0):
                extra_struct=struct[abs(min(0,xmov[0])):min(structsize,self.DIM-xmov[0]),0:abs(ymov[0]),abs(min(0,zmov[0])):min(structsize,self.DIM-zmov[0])]
                full_struct[xmin:xmax,self.DIM-abs(ymov[0]):self.DIM,zmin:zmax]=np.add(full_struct[xmin:xmax,self.DIM-abs(ymov[0]):self.DIM,zmin:zmax],extra_struct)
            if (ymov[1]>self.DIM):
                extra_struct=struct[abs(min(0,xmov[0])):min(structsize,structsize+self.DIM-xmov[1]),structsize-(ymov[1]-self.DIM):structsize,abs(min(0,zmov[0])):min(structsize,structsize+self.DIM-zmov[1])]
                full_struct[xmin:xmax,0:ymov[1]-self.DIM,zmin:zmax]=np.add(full_struct[xmin:xmax,0:ymov[1]-self.DIM,zmin:zmax],extra_struct)
            if (zmov[0]<0):
                extra_struct=struct[abs(min(0,xmov[0])):min(structsize,self.DIM-xmov[0]),abs(min(0,ymov[0])):min(structsize,self.DIM-ymov[0]),0:abs(zmov[0])]
                full_struct[xmin:xmax,ymin:ymax,self.DIM-abs(zmov[0]):self.DIM]=np.add(full_struct[xmin:xmax,ymin:ymax,self.DIM-abs(zmov[0]):self.DIM],extra_struct)
            if (zmov[1]>self.DIM):
                extra_struct=struct[abs(min(0,xmov[0])):min(structsize,structsize+self.DIM-xmov[1]),abs(min(0,ymov[0])):min(structsize,structsize+self.DIM-ymov[1]),structsize-(zmov[1]-self.DIM):structsize]
                full_struct[xmin:xmax,ymin:ymax,0:zmov[1]-self.DIM]=np.add(full_struct[xmin:xmax,ymin:ymax,0:zmov[1]-self.DIM],extra_struct)
        
    #truncated struct if some part is outside the full struct
        small_struct=struct[abs(xmov[0]-xmin):structsize-abs(xmov[1]-xmax), abs(ymov[0]-ymin):structsize-abs(ymov[1]-ymax),abs(zmov[0]-zmin):structsize-abs(zmov[1]-zmax)]
        #add to full box
        full_struct[xmin:xmax,ymin:ymax,zmin:zmax] = np.add(full_struct[xmin:xmax,ymin:ymax,zmin:zmax],small_struct) 

        return full_struct

