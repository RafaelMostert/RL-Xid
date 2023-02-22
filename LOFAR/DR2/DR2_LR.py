#!/usr/bin/env python3

# coding: utf-8

# ## Likelihood Ratio for DR2 ##
# 
# This notebook has the running of the Likelihood Ratio Code on DR2.  Once the output from the Ridgeline Code has been produced this code can be used to determine the list of possible hosts for the DR2 data set.

# In[ ]:


# Imports
import numpy as np
from astropy.io import fits
import os
from astropy.table import Table
import pandas as pd
from os.path import exists
from sklearn.neighbors import KernelDensity
import RidgelineFilesDR2 as RLF
import RLConstantsDR2 as RLC
from ridge_toolkitDR2 import DefineCutoutHDU, GetAvailableSources, GetCutoutArray
from SourceSearchDR2 import GetSourceList, SourceInfo, filter_sourcelist, CreateLDistTable, NClosestDistances, GetCatArea, TableFromLofar, CreateSubCat, CreateCutOutCat, LikelihoodRatiosLognormal
import sys
from astropy_healpix import HEALPix
import astropy.units as u

overwrite = bool(int(os.getenv('PIPE_OVERWRITE')))
if os.path.exists(RLF.PossHosts) and not overwrite:
    print("DONE: Calculated ridgeline likelihood ratiosbeen.")
    exit()
else:
    print("LR Debug:", RLF.PossHosts, overwrite)
hp = HEALPix(nside=16)

fitsfile=RLF.TFC.replace('.txt','.fits')
available_sources=Table.read(fitsfile)

# Load in the optical/IR and LOFAR catalogues, form tables and df and save as text files 
# Check columns esp on the Opt/IR
#OptTable = TableOfSources(str(RLF.OptCat))
OptTable = Table.read(RLF.OptCat)
# There aren't zillions of columns so fine to use whole thing
LofarTable = TableFromLofar(RLF.LofCat)
Lofardf = LofarTable.to_pandas()
Optdf = OptTable.to_pandas()
#Optdf.to_csv(RLF.OptCatdf, columns = [str(RLF.IDW), str(RLF.IDP), str(RLF.PossRA), str(RLF.PossDEC), str(RLF.OptMagA), str(RLF.OptMagP)], header = True, index = False)



# Set up source_list to be used in all of the following fuctions/cells
probfile = RLF.psl
source_list = GetSourceList(available_sources, probfile)

# source_list is now a filtered table

zerococ=[]
for source in source_list:
    source_name = source['Source_Name']
    lofarra = source['RA']
    lofardec = source['DEC']
    sizepix = source['Size']

    size = sizepix * RLC.ddel # convert size in pixels to degrees
    subcat = Optdf[(np.abs(Optdf[str(RLF.PossRA)] - lofarra) * np.cos(lofardec * np.pi / 180.0) < size) & (np.abs(Optdf[str(RLF.PossDEC)] - lofardec) < size)].copy()

    # Insert the uniform optical position error if required             

    subcat['raErr'] = np.where(np.isnan(subcat[str(RLF.OptMagP)]), RLC.UniWErr, RLC.UniLErr)
    subcat['decErr'] = np.where(np.isnan(subcat[str(RLF.OptMagP)]), RLC.UniWErr, RLC.UniLErr)

    subcat.to_csv(RLF.MagCO %source_name, columns = [str(RLF.IDW), str(RLF.IDP), str(RLF.PossRA), str(RLF.PossDEC), 'raErr', 'decErr', str(RLF.OptMagA), str(RLF.OptMagP)], header = True, index = False)
    print("Subcat written for source",source_name)

            

# Looping through all successful sources to create the cutoutcat .txt files.  The distance away to form
# the sub-catalogue is set in RLConstants and is currently set to 1 arcmin RA and 0.5 arcmin DEC.
# Only needs to be done once

source_count = 0 ## Keeps track of where the loop is
#print("Length of source list before COC loop: ",len(source_list))

for source in source_list:
    source_name=source['Source_Name']
    size = source['Size']
    #lofar_ra, lofar_dec = SourceInfo(source, LofarTable)[:2] # why was this needed?
    lofar_ra = source['RA']
    lofar_dec = source['DEC']

    subcat1 = CreateSubCat(OptTable, lofar_ra, lofar_dec)
    print('Optical subcat length is',len(subcat1))
    
    # Insert the uniform optical position error if required
    subcatdf = subcat1.to_pandas()

    # Insert the uniform optical position error if required             

    subcatdf['raErr'] = np.where(np.isnan(subcatdf[str(RLF.OptMagP)]), RLC.UniWErr, RLC.UniLErr)
    subcatdf['decErr'] = np.where(np.isnan(subcatdf[str(RLF.OptMagP)]), RLC.UniWErr, RLC.UniLErr)
    subcat2 = Table.from_pandas(subcatdf)
    #print("About to get COC for ",source)
    cutoutcat,lcat = CreateCutOutCat(source_name, LofarTable, subcat2, lofar_ra, lofar_dec, size)
    source_count += 1
    print("Source Number = ",source_count)
    if lcat==0:
        print("Removing source for zero-length cutout cat: ",source_name)
        zerococ.append(source_name)

source_list=filter_sourcelist(source_list,zerococ)
        
print('Number of viable sources = ',len(source_list))

if len(source_list)==0:
    sys.exit(0)

# Create a table of R distance information from LOFAR Catalogue position
# Only needs to be done once - need to exclude probs?
for source in source_list:
    print("Creating distance table for:",source['Source_Name'])
    CreateLDistTable(source['Source_Name'],source_list)


# Find the 30 closest sources for each ridgeline
# Only needs to be done once
n = 30
print('Finding closest distances')
NClosestDistances(source_list, LofarTable, n)

#area=GetCatArea(RLF.OptCat)

hpix=hp.lonlat_to_healpix(OptTable['RA']*u.deg,OptTable['DEC']*u.deg)

uhp=sorted(np.unique(hpix))
print('There are',len(uhp),'healpixes in the optical catalogue')
area=hp.pixel_area.to(u.arcsec*u.arcsec).value

print('Catalogue area is',area,'sq. arcsec')

# Generating the likelihood ratios for all possible close sources, for each drawn
# ridgeline, using the R distance from LOFAR Catalogue position and the ridgeline.
# Only needs to be done once
print('Finding LRs. Using lognormal form of f(r).')
LikelihoodRatiosLognormal(source_list)

################################### RM: speed hack remove when done comparing ridge dists
#print("WARNING EARLY STOPPING TO CHECK RIDGE SIZES! in DR2_LR.py ADJUST FOR REGULAR RUN!")
#sys.exit(0)
# CREATE NEAREST 30 INFO
# Load in the three text files for each source, join all the table information together and save
# Only needs to be done once

for asource in source_list:
    source=asource['Source_Name']
    print('Creating nearest 30 tables for',source)

    combined_fs = pd.read_csv(RLF.LLR %source, header = 0)
    #LofarLR = pd.read_csv(RLF.LLR %source, header = 0)
    #RidgeLR = pd.read_csv(RLF.RLR %source, header = 0, usecols = ['Ridge_LR'])
    MagCutOut = pd.read_csv(RLF.MagCO %source, header = 0, usecols = [RLF.IDW, RLF.IDP, RLF.PossRA, RLF.OptMagA, RLF.OptMagP])
    MagCutOut[str(RLF.PossRA)] = MagCutOut[str(RLF.PossRA)].apply(lambda x: round(x, 7))
            
    All_LR = combined_fs
    # Changed combined to use just the lofar value if the ridge value is nan
        
    All_LR.columns=['combined_distance', 'combined_f', str(RLF.PossRA), str(RLF.PossDEC)]
    All_LR[str(RLF.PossRA)] = All_LR[str(RLF.PossRA)].apply(lambda x: round(x, 7))
            
    MagLR = All_LR.merge(MagCutOut, on = str(RLF.PossRA))
            
    MagLR.to_csv(RLF.LRI %source, columns = ['combined_distance', 'combined_f', str(RLF.PossRA), str(RLF.PossDEC), str(RLF.IDW), str(RLF.IDP), str(RLF.OptMagP), str(RLF.OptMagA)], header = True, index = False)


print('Colour LR calculations')


# Load in parent population host info
Hosts = pd.read_csv(str(RLF.DR1Hosts), usecols = [ 'Source_Name', 'AllWISE', 'Host_RA', 'Host_DEC', 'W1mag', 'i'], header = 0)
Hosts['r'] = Hosts['i'].apply(lambda y: y + 0.33)
Hosts['Colour'] = Hosts['r'].astype(np.float64).subtract(Hosts['W1mag'].astype(np.float64), axis = 'index')


# In[47]:


# Create the colour column and sample the df
#pwfulldf = pd.read_csv(RLF.OptCatdf, header = 0, usecols = [RLF.IDW,RLF.OptMagP,RLF.OptMagA])
pwfulldf=Optdf # already in memory
# Temporary update to keep in WISE-only galaxies
ColourPW = pwfulldf[~np.isnan(pwfulldf[RLF.OptMagP]) & ~np.isnan(pwfulldf[RLF.OptMagA])].copy()
ColourPW.reset_index(drop = True, inplace = True)
ColourPW['Colour'] = ColourPW[RLF.OptMagP].subtract(ColourPW[RLF.OptMagA], axis = 'index')
#ColSam = ColourPW.sample(50000, replace = False)
ColSam = ColourPW.sample(10000, replace = False) # RM: lowered sample for ~8xspeed improvement


'''
# Skyarea Covered byt he LOFAR data set
area = (np.deg2rad(RLC.LRAu) - np.deg2rad(RLC.LRAd)) * (np.sin(np.deg2rad(RLC.LDECu)) - np.sin(np.deg2rad(RLC.LDECd))) * np.rad2deg(3600)**2
'''


# Not running on i-band so only need the W1 band cells


h_train = np.vstack([Hosts['W1mag'], Hosts['Colour']]).T
kde_h = KernelDensity(kernel = 'gaussian', bandwidth = RLC.bw)
kde_h.fit(h_train)

if RLC.norm_q:
    hh, ww1 = np.mgrid[Hosts['Colour'].min() : Hosts['Colour'].max() : 0.05, Hosts['W1mag'].min() : Hosts['W1mag'].max() : 0.05]
    h_sample = np.vstack([ww1.ravel(), hh.ravel()]).T
    prob_h = np.exp(kde_h.score_samples(h_sample))
    norm_q = len(Hosts['W1mag'])/np.sum(prob_h)
else:
    norm_q = 1

o_train = np.vstack([ColSam[RLF.OptMagA], ColSam['Colour']]).T
kde_o = KernelDensity(kernel = 'gaussian', bandwidth = RLC.bw)
kde_o.fit(o_train)


if RLC.norm_n:
    oo, ww2 = np.mgrid[ColSam['Colour'].min() : ColSam['Colour'].max() : 0.05, ColSam[RLF.OptMagA].min() : ColSam[RLF.OptMagA].max() : 0.05]
    o_sample = np.vstack([ww2.ravel(), oo.ravel()]).T
    prob_o = np.exp(kde_o.score_samples(o_sample))
    norm_n = len(ColSam[RLF.OptMagA])/np.sum(prob_o)
else:
    norm_n = 1


def GetLR(fr, qm, nm):
    lr = (fr * qm) / nm
    return lr

def GetLR2(fr, qm, nm,debug=False):
    if debug: print("fr, qm and nm are: ",fr,qm,nm)
    # we want density in units of per typical source length
    # squared for LOFAR f(R) and sqrt(density) in arcsec
    #for RL f(R) - take 60 arcsec
    lr = (fr * qm) / ((nm**1.5)/(np.sqrt(area)*area/(60.0*60.0)))
    return lr

def Getqmc(m, c, model_error,norm=1):
    qmc = np.exp(kde_h.score_samples(np.array([m, c]).reshape(1, -1)))
    norm_qmc = qmc * norm + model_error # RM: the model_error roughly 10% of max q(m,c) is for regularisation
    print(f"q(m,c)=q({m},{c})={norm_qmc}")
    return norm_qmc

def Getnmc(m, c, model_error,norm=1):
    nmc = np.exp(kde_o.score_samples(np.array([m, c]).reshape(1, -1)))
    norm_nmc =  nmc * norm + model_error # RM: the model_error roughly 10% of max n(m,c) is for regularisation
    print(f"n(m,c)=n({m},{c})={norm_nmc}")
    return norm_nmc

def Getqmc_unregularised(m, c, norm=1):
    qmc = np.exp(kde_h.score_samples(np.array([m, c]).reshape(1, -1)))
    return qmc * norm

def Getnmc_unregularised(m, c, norm=1):
    nmc = np.exp(kde_o.score_samples(np.array([m, c]).reshape(1, -1)))
    return nmc * norm

# Roughly determine the max qmc and nmc values
def estimate_maxima(no_samples = 100, q_m_min=16, q_m_max=20, q_c_min=1, q_c_max=3, 
                    n_m_min=19, n_m_max=22, n_c_min=0, n_c_max=3):
    """Return estimate of maximum qmc and nmc values. 
    These values can later be used for regularisation.
    """
    m_samples = np.linspace(q_m_min,q_m_max, no_samples)
    c_samples = np.linspace(q_c_min,q_c_max, no_samples)
    q_max = np.array([Getqmc_unregularised(m,c, norm=norm_q) for m,c in zip(m_samples, c_samples)]).max()
    m_samples = np.linspace(n_m_min,n_m_max, no_samples)
    c_samples = np.linspace(n_c_min,n_c_max, no_samples)
    n_max = np.array([Getnmc_unregularised(m,c, norm=norm_n) for m,c in zip(m_samples, c_samples)]).max()
    print(f"We find q_max ={q_max:.3f}, and n_max ={n_max:.3f}.")
    return q_max, n_max

# Get regularisation factors
q_max, n_max = estimate_maxima(no_samples = 100, q_m_min=16, q_m_max=20, q_c_min=1, q_c_max=3, 
                    n_m_min=19, n_m_max=22, n_c_min=0, n_c_max=3)

# Calculating the LR from the text files for the W1 band hosts

# Removed the line where you deal with taking the W1 value if the r band value was 0.

# try just fixing the colours here

newdrop=[]
for asource in source_list:
    source=asource['Source_Name']
            
    MLR = pd.read_csv(str(RLF.LRI) %source, header = 0, usecols = ['combined_distance', 'combined_f', str(RLF.PossRA), str(RLF.PossDEC), str(RLF.IDW), str(RLF.IDP), str(RLF.OptMagP), str(RLF.OptMagA)])
    MLR['Col'] = MLR[RLF.OptMagP].subtract(MLR[RLF.OptMagA], axis = 'index')
    MLR['Colour'] = MLR.apply(lambda row: np.where(row[RLF.OptMagP]>0.0,row[RLF.OptMagP]-row[RLF.OptMagA],RLC.meancol),axis=1).astype(np.float128)
    MCLT = MLR[~np.isnan(MLR['Colour'])].copy()
    MCLR = MCLT[~np.isnan(MLR[RLF.OptMagA])].copy()
    print("Source is",source,"and MCLR has length",len(MCLR))
    if(len(MCLR)<1):
        newdrop.append(source)
    else:
        
        # Regularisation factor is 0.1 times the max value of q or n
        MCLR[str(RLF.LRMC)] = MCLR.apply(lambda row: GetLR2(row['combined_f'],
            Getqmc(row[RLF.OptMagA], row['Colour'], 0.1*q_maxm, norm=norm_q), Getnmc(row[RLF.OptMagA], 
            row['Colour'], 0.1*n_max, norm=norm_n)), axis = 1).astype(np.float128)
                
        MCLR.to_csv(str(RLF.LR) %source, columns = ['combined_distance', str(RLF.PossRA), str(RLF.PossDEC), str(RLF.IDW), str(RLF.IDP), str(RLF.OptMagP), str(RLF.OptMagA), str(RLF.LRMC)], header = True, index = False)


source_list=filter_sourcelist(source_list,newdrop)
        
# For each source in the list find the maximum combined LR and store all the information

def FindMax(source):
    info = pd.read_csv(RLF.LR %source, header = 0, usecols = [str(RLF.PossRA), str(RLF.PossDEC), str(RLF.IDW), str(RLF.IDP), str(RLF.LRMC)])
    info[str(RLF.IDW)] = info[str(RLF.IDW)].map(lambda x: x.strip('b').strip("''"))
    info[str(RLF.IDP)] = info[str(RLF.IDP)].map(lambda x: x.strip('b').strip("''"))
    #info[str(RLF.ID3)] = info[str(RLF.ID3)].map(lambda x: x.strip('b').strip("''"))
    CP = info.loc[info[str(RLF.LRMC)].idxmax()].copy()
    CP['PossFail'] = np.where(CP[str(RLF.LRMC)] < RLC.Lth, 1, 0)
    CP[str(RLF.LSN)] = source
    
    return CP

print('Finding max LRs and writing table')

PossHosts = pd.concat([FindMax(s['Source_Name']) for s in source_list], ignore_index = True, axis = 1)
PossHosts.columns = PossHosts.loc[str(RLF.LSN)]
PossHosts = PossHosts.drop(index = [str(RLF.LSN)])
PossHostsT = PossHosts.transpose()
print("Writing hosts.csv file to:", RLF.PossHosts)
PossHostsT.to_csv(RLF.PossHosts, header = True, index = True)#,  columns = [str(RLF.LSN), str(RLF.PossRA), str(RLF.PossDEC), str(RLF.IDW), str(RLF.IDP), str(RLF.ID3), str(RLF.OptMagP), str(RLF.OptMagA)]


# In[ ]:


# Number of sources with a 0 max LR and therefore would possibly be a failed LR
# or defined by being closest to LOFAR
print('Total number of fails is',np.sum(PossHostsT['PossFail']))

