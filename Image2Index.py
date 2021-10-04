# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 18:12:09 2021

@author: Ámbar Pérez García
"""
import numpy as np
import spectral.io.envi as envi
import json
import matplotlib.pyplot as plt

def Image2Index(path, sensor):
    # Open image
    data = envi.open(path + '.hdr', path + '.bin')
    image = data.open_memmap(writeable = True)
    
    # Load bands
    if sensor == "AVIRIS":
        # FI
        R = image[:,:,28] # AVIRIS (band 29 - 638nm)
        G = image[:,:,19] # AVIRIS (550nm)
        B = image[:,:,13] # AVIRIS (491nm)
        # HI
        Ra = image[:,:,141] # AVIRIS (1701nm)
        Rb = image[:,:,144] # AVIRIS (1731nm)
        Rc = image[:,:,145] # AVIRIS (1741nm)
        # OSI
        R675 = image[:,:,34] # AVIRIS (675nm)
        R743 = image[:,:,41] # AVIRIS (743nm)
        # RAI
        R889 = image[:,:,56] # AVIRIS (889nm)
        # CDOM
        R565 = image[:,:,20] # AVIRIS (560nm)
        R660 = image[:,:,30] # AVIRIS (657nm)
        # CHL
        R433 = image[:,:,7] # AVIRIS (433nm)
        R490 = image[:,:,13] # AVIRIS (591nm)
        R510 = image[:,:,15] # AVIRIS (511nm)
        R555 = image[:,:,20] # AVIRIS (560nm)
        # B2/B11 Sentinel2
        R1610 = image[:,:,132] # AVIRIS (1612nm)
        # NDWI
        R860 = image[:,:,53] # AVIRIS (860nm)
        R1240 = image[:,:,93] # AVIRIS (1244nm)
        """# RG
        R501 = image[:,:,14] # AVIRIS (501nm)
        R521 = image[:,:,16]; R530 = image[:,:,17] 
        R540 = image[:,:,18]; R569 = image[:,:,21] 
        # RR
        R618 = image[:,:,26] # AVIRIS (618nm)
        R628 = image[:,:,27]; R647 = image[:,:,29]; R667 = image[:,:,31]
        R655 = image[:,:,32]; R665 = image[:,:,33]; R685 = image[:,:,35]
        R694 = image[:,:,36]; R704 = image[:,:,37]; R714 = image[:,:,38]
        R724 = image[:,:,39]; R734 = image[:,:,40]; R753 = image[:,:,42]
        R763 = image[:,:,43]; R773 = image[:,:,44]"""
        # WAF
        R1343 = image[:,:,105] # AVIRIS (1343nm)
        R1563 = image[:,:,127] # AVIRIS (1562nm)
        R1453 = image[:,:,116] # AVIRIS (152nm)
        # NDOI
        R599 = image[:,:,25] # AVIRIS (599nm)
        R870 = image[:,:,55] # AVIRIS (870nm)
        
        ###############################################################################################
        # FI
        Av_FI = (B.astype(float) - R.astype(float))/(B + R)
        # HI
        Av_HI = (1731-1701)*(Rc.astype(float) - Ra.astype(float))/(1741-1701) + Ra.astype(float) - Rb.astype(float)
        # OSI
        Av_OSI = (R743.astype(float) - R675.astype(float))/(748-678)
        # RAI
        modulo = np.sqrt(B.mean()**2 + R889.mean()**2) # Approach
        Av_RAI = modulo*(B.astype(float) - R889.astype(float))/(B + R889)
        # CDOM
        Av_CDOM = R565.astype(float)/R660.astype(float)
        # CHL
        Av_CHL = np.log(abs(max(R433.mean(),R490.mean(),R510.mean())/R555.astype(float)))
        # NDVI
        Av_NDVI = (R889.astype(float) - R.astype(float))/(R889 + R)
        # B2/B11 Sentinel2
        Av_S211 = B.astype(float)/R1610
        # NDWI
        Av_NDWI = (R860.astype(float) - R1240.astype(float))/(R860 + R1240)
        # WAF
        Av_WAF = (R1343.astype(float) - R1563.astype(float))/2 - R1453
        # NDOI
        Av_NDOI = (R599.astype(float) - R870.astype(float))/(R599 + R870)
        
        indexes = np.array([Av_NDOI, Av_RAI, Av_FI, Av_HI, Av_OSI, Av_WAF, Av_CDOM, Av_CHL, Av_NDVI, Av_NDWI, Av_S211])
        index_name = ["NDOI", "RAI", "FI", "HI", "OSI", "WAF", "CDOM", "CHL", "NDVI", "NDWI", "B2/B11"]
    
    elif sensor == "HICO":
        # FI
        R = image[:,:,42] # HICO (639nm)
        G = image[:,:,27] # HICO (553nm)
        B = image[:,:,11] # HICO (461nm)
        # OSI
        R675 = image[:,:,55] # HICO (667nm)
        R743 = image[:,:,70] # HICO (753nm)
        # RAI
        R889 = image[:,:,93] # HICO (885nm)
        # CDOM
        R565 = image[:,:,37] # HICO (564nm)
        R660 = image[:,:,54] # HICO (662nm)
        # CHL
        R433 = image[:,:,14] # HICO (433nm)
        R490 = B
        R510 = image[:,:,28] # HICO (513nm)
        R555 = image[:,:,35] # HICO (553nm)
        # NDOI
        R599 = image[:,:,43] # HICO (599nm)
        R870 = image[:,:,91] # HICO (873nm)
        
        ###############################################################################################
        # FI
        Hico_FI = (B - R)/(B + R)
        # OSI
        Hico_OSI = (R743 - R675)/(748-678)
        # RAI
        modulo = np.sqrt(B.mean()**2 + R889.mean()**2)
        Hico_RAI = modulo*(B - R889)/(B + R889)
        # CDOM
        Hico_CDOM = R565/R660
        # CHL
        Hico_CHL = np.log(max(R433.mean(),R490.mean(),R510.mean())/R555)
        # NDVI
        Hico_NDVI = (R889.astype(float) - R.astype(float))/(R889 + R)
        # NDOI
        Hico_NDOI = (R599.astype(float) - R870.astype(float))/(R599 + R870)
        
        indexes = np.array([Hico_NDOI, Hico_RAI, Hico_FI, Hico_OSI, Hico_CDOM, Hico_CHL, Hico_NDVI])
        index_name = ["NDOI", "RAI", "FI", "OSI", "CDOM", "CHL", "NDVI"]
    
    elif sensor == "MERIS":
        #  FI
        R = image[:,:,6] # MERIS (665nm)
        G = image[:,:,4] # MERIS (560nm)
        B = image[:,:,2] # MERIS (490nm)        
        # OSI
        R753 = image[:,:,9] # MERIS (753nm)
        R665 = R
        # RAI
        R885 = image[:,:,12] # MERIS (885nm) B14       
        # CDOM
        R565 = G # MERIS (560nm)
        R660 = image[:,:,5] # MERIS (620nm) 
        # CHL
        R433 = image[:,:,1] # MERIS (442nm)
        R490 = B
        R510 = image[:,:,3] # MERIS (510nm)
        R555 = G
        # NDOI
        R599 = G 
        R870 = image[:,:,11] # MERIS (865nm) B13
        
        ###############################################################################################
        # FI
        Env_FI = (B[:,:] - R[:,:])/(B[:,:] + R[:,:])
        # OSI
        Env_OSI = (R753[:,:] - R665[:,:])/(748-678)
        # RAI
        modulo = np.sqrt(B[:,:].mean()**2 + R885[:,:].mean()**2)
        Env_RAI = modulo*(B[:,:] - R885[:,:])/(B[:,:] + R885[:,:])
        # CDOM
        Env_CDOM = R565[:,:]/R660[:,:]
        # CHL
        Env_CHL = np.log(max(R433[:,:].mean(),R490[:,:].mean(),R510[:,:].mean())/(R555[:,:]))
        # NDVI
        Env_NDVI = (R885[:,:] - R[:,:])/(R885[:,:] + R[:,:])
        # NDOI
        Env_NDOI = (R599[:,:] - R870[:,:])/(R599[:,:] + R870[:,:])
        
        indexes = np.array([Env_NDOI, Env_RAI, Env_FI, Env_OSI, Env_CDOM, Env_CHL, Env_NDVI])
        index_name = ["NDOI", "RAI", "FI", "OSI", "CDOM", "CHL", "NDVI"]
    
    else: print("Sensor not available.")

    # RGB
    """
    from PIL import Image
    rgb = np.stack([R, G, B], axis=-1); rgb = rgb/rgb.max()        
    im = Image.fromarray((rgb*255).astype(np.uint8), 'RGB')
    im.save(path + ".jpeg") """
    """# Meris 
    t=fix_contrast(R870)
    im = Image.fromarray((t/t.max()*255).astype(np.uint8))"""
    
    return indexes, index_name

def get_labels(path): # Get labels from ROIs
    pixels = []
    y = np.array([])
    
    with open(path + ".json", "r") as read_file: # get ROIs
        data = json.load(read_file)

    for i in data: # List of all pixel selected
        pixels += data[i]
        if not y.size: y = np.repeat(i, len(data[i]))
        else: y = np.append(y, np.repeat(i, len(data[i])))
    
    pixels = [int(i) for i in pixels]
    classes = [i for i in data] # Names of classes
    
    return pixels, classes, y

def get_values(pixels, indexes): # Get selected pixel values from all indexes
    nvector = np.array([[]])
    values = np.array([[]])
    
    for j in range(0, indexes.shape[0]):
        for i in pixels:
            flat = indexes[j].flatten()
            if not nvector.size: nvector = np.array([[flat[i]]])
            else: nvector = np.append(nvector, np.array([[flat[i]]]), axis=0)
        
        if not values.size: values = nvector
        elif values.ndim == 2: values = np.append(np.array([values]), np.array([nvector]), axis=0)
        else: values = np.append(values, np.array([nvector]), axis=0)
        nvector = np.array([[]])
    
    return values

def histogram(X, sensor, index, represent = None, Y=None, classes = None, bins = 35):
    if represent == "Simple":
        plt.hist(X, color='#F2AB6D', rwidth=0.85)
        plt.title(sensor)
        plt.xlabel(index)
        plt.ylabel('Frecuencia')

    if represent == "Stack":
        kwargs = dict(alpha=0.5, bins=bins, density=True, stacked=True)
        for clas in classes:
            plt.hist(X[Y == clas], **kwargs, label=clas)
        plt.title(sensor)
        plt.xlabel(index)
        plt.ylabel('Frecuencia')
        plt.legend(prop={'size': 10})
    
    
    

    
    
