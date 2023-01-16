# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 16:29:04 2022

@author: sueco
"""

########     Annimated line plot


# Import libraries

from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt 
import numpy as np

# Create the fugure and axes objections
#fig,ax = plt.subplots()

# Create empty lists for the x and y data
x = [0]
y = [0]

n=10000 # number of frames
sd = 2

#define the data
x1 = np.random.normal(0, sd, 10000)
x2 = np.random.gamma(2, 1.5, 10000)-sd*2
x3 = np.random.exponential(sd*2, 10000)-sd*2
x4 = np.random.uniform(0,sd*4, 10000)-sd*2


# Make the subplots and set plotting params
fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2, sharex=True, sharey=True,figsize=(15, 15))

bins = np.arange(-sd*4, sd*4,0.5)

#put both axes and data into lists to itterate through
axs = [ax1,ax2,ax3,ax4]
xs = [x1,x2,x3,x4]

titles = ['Normal','Gamma','Expenential','Uniform']


#Function that draws each frame of the animation

def update(i):
    #Check if animation is at the last frame, and if so, stop the animation 
    if i==n/skip:
        a.event_source.stop()
        
    #For each axis - plot the corresponding data
    # should do this as a dictionary.....
    for j in range(0,len(axs)):
        axs[j].cla()
        axs[j].spines['right'].set_visible(False)
        axs[j].spines['top'].set_visible(False)
        axs[j].hist(xs[j][:skip*i],bins=bins,color='gray')
        axs[j].set_ylabel('Frequency')
        axs[j].set_xlabel('Value')
        axs[j].set_title(titles[j])
        if j==0:
            ax1.axis([-sd*4,sd*4,0,n/6])
            axs[j].annotate('n = {}'.format(i*skip), [-sd*2,n/8])

#run the animation
skip=150
a = FuncAnimation(fig, update,  interval=100, repeat = False, save_count = n/skip)

#save the animation as a gif
f = r"DistributionAnimation.gif" 
writergif = PillowWriter(fps=30) 
a.save(f, writer=writergif)