import numpy as np
import matplotlib.pyplot as plt
from Random_bubbles import *
from scipy.special import jv  
plt.ion()
h=0.6774
colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c','#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5','#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f','#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

def W(y):
	# return jv(1,y)/y
	return 3 / y**3 * (np.sin(y)-y*np.cos(y))

""" computes the power spectrum for a 3D fully symmetric box, no periodic boundary conditions """
NDIM=3
L = 128./h#Mpc
N = 512#sampling
deltax = L / N
maxk = 0.5* 2.*np.pi/deltax #Nyquist-Shannon

f = 0.001 # filling fraction
radii = np.array([5,10]) # test different radius sizes
nbs = np.array(3.* N**3 * f / 4. / np.pi / radii**3, dtype=int) # number of bubbles required for given filling fraction

#k-grid
k_x=np.fft.fftfreq(N)*2*np.pi*N/L #gives frequency spectrum corresponding to the Fourier decomposition for N1 cells
a=np.power(k_x,2)[:, None] + np.power(k_x,2) 
b=a[:,:,None] + np.power(k_x,2)
k_norm=np.sqrt(b)

nbins=65 # spherical bins
delta_k=2*np.pi/L
k_bin=np.logspace(np.log10(delta_k),np.log10(maxk),nbins)
kmid=0.5*(k_bin[:-1]+k_bin[1:]) #centre of each bin, nbk-1 elements

ks = np.logspace(-2,1,100)  
PSs = np.zeros((nbins-1,radii.size+2))
PSs[:,0]=kmid
for u in range(0,radii.size):

	print('\nGenerating box')
	cube = RandomBubbles(DIM=N,nb=nbs[u],radius=radii[u],NDIM=NDIM)
	field = cube.box

	print('Computing FT')
	delta = field/np.mean(field) -1.
	#Fourier transform
	delta_k=np.fft.fftn(field,norm='ortho') #normalisation is 1/sqrt(N)
	#power spectrum
	PS=np.real(delta_k*np.conjugate(delta_k))
	print('Binning PS')
	P_store=np.zeros(nbins-1)
	k_store=np.zeros(nbins-1)
	for m in range(0,nbins-1):
	    mask= (k_bin[m]<=k_norm) & (k_norm<k_bin[m+1])
	    if np.any(mask): #cannot compute mean if zero elements
	        P_store[m]=np.mean(PS[mask])
	        k_store[m]=np.sum(mask)
	P_store = P_store/(N/L)**3
	PSs[:,u+1] = P_store

	r = radii[u]*L/N
	plt.figure()
	plt.loglog(kmid[k_store!=0],P_store[k_store!=0],marker='o',markersize=3,color=colors[int(2*u)],lw=0)
	plt.loglog(ks, f * W(ks*r)**2 * (4./3.*np.pi*r**3),color=colors[int(2*u)],ls='-') 
	plt.loglog(ks, f * (3/ks**2/r**2)**2 * (4./3.*np.pi*r**3),color='k',ls='--',lw=1.)
	plt.axhline(1,color='k',lw=.8)
	plt.axhline(f * (4./3.*np.pi*r**3),color='k',ls='--',lw=1.)
	plt.axvline(9**(1/4)/r,color='k',ls='--',lw=1.)  
	plt.axvline(1/r,color='k',ls=':')
	plt.xlabel(r'$k$ [Mpc$^{-1}$]')
	plt.ylabel(r'$\mathcal{P}(k)$ [Mpc$^3$]')
	plt.xlim(1e-2,10)
	plt.tight_layout()
	plt.savefig('Bubble_power_spectrum_3d_R%i.png' %(radii[u]))
	plt.close()

