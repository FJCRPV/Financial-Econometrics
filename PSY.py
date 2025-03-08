import numpy as np
from ADF import ADF

def PSY(y, swindow0="",IC=0,adflag=0):
    t = len(y)
    if swindow0 == "":
        swindow0 = np.floor(t*(0.01+1.8/np.sqrt(t)))
        print(swindow0)
    swindow0 = float(swindow0)
    bsadfs = np.full((t,1), None)
    for r2 in range(int(swindow0), t+1):
        rwadft = np.full((r2-int(swindow0)+1, 1),-999.0)
        for r1 in range(1, r2-int(swindow0) + 1 + 1):
            rwadft[r1-1] = ADF(y[r1-1:r2], IC, adflag)
        bsadfs[r2-1,0] = max(np.array(rwadft).flatten())

    bsadf = bsadfs[int(swindow0)-1:t]
    return bsadf.flatten()

