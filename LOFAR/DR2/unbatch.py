#!/usr/bin/python

import numpy as np
from astropy.table import Table,vstack
import sys
import os
import glob

root=sys.argv[1]
outfile=root+"_allhosts.csv"
outfits=root+"_allhosts.fits"
outrand=root+"_randhosts.fits"

files=glob.glob(root+"*/hosts.csv")
print("Unbatching ridgeline output:", files)
lout=open(outfile,"w")
lout.write("Source_Name,RA,DEC,UNWISE_OBJID,UID_L,LRMagBoth,ID_R\n")
for d in files:
    lines=[l.rstrip() for l in open(d).readlines()]
    lines=lines[1:]
    #print lines +"\n"
    for l in lines:
        lout.write(l+"\n")
    #print d

lout.close()
    
# make random test sample of sample size


intab=Table.read(outfile,format='ascii',delimiter=',')
#gtab=intab[intab['LRMagBoth']>1.0]
gtab=intab # Commented out the line above
sampsize = min(100, len(gtab))

#nums=np.random.choice(len(gtab),sampsize,replace=False)
#keep=gtab[nums]
#keep.write(outrand,overwrite=True)

gtab.remove_columns(['UID_L'])
gtab.rename_column('RA','optRA_RLC')
gtab.rename_column('DEC','optDEC_RLC')

gtab.write(outfits,overwrite=True)
#print("And a fits version written to:", outfile)
#print("Random test sample written to:", outrand)

#print intab['LRMagBoth']
print("Writing resulting combined file to:", outfile)
print("All DONE.")
