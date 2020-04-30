#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:44:15 2020

@author: Brenda
"""

def seg2labels(method,image):
    '''

    Parameters
    ----------
    method : string
    image : ndarray

    Returns
    -------
    img_slic : ndarray of segments
    img_slic_bounds : ndarray of segment boundaries

    '''
    from skimage.segmentation import slic, mark_boundaries
    if method == 'slic':
       # print('slic segmentation goes here')
        base_im = image
        img_slic = slic(base_im,n_segments=25)
        img_slic_bounds = mark_boundaries(base_im,img_slic,color=(1,0,0))
        return (img_slic,img_slic_bounds)
    elif method == 'felzenswab':
        print('felzenswab segmentation goes here')   
    elif method == 'quickshift':
        print('quickshift segmentation goes here')
    else: print('method not recognized') 