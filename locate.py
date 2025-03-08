import numpy as np

def locate (index, dates):
    maxi = np.amax(index)
    lc = np.where(index == np.amax(index)) 
    
    if maxi == 1: # there is at least 1 episode
        count = 0
        EP = np.zeros((30,2), dtype = object)
        # maximum 20 episodes: col1 origination date col2 termination date
        i = lc[0][0] + 1
        EP[count, 0] = dates[i-2]
        while i <= len(index):
            if (index[i-2] == 1) & (index[i-1] == 0):
                EP[count, 1] = dates[i-2] # termination date
                i += 1
            elif (index[i-2] == 0) & (index[i-1] == 1):
                count += 1
                EP[count, 0] = dates[i-1] # origination date
                i += 1
            else:
                i += 1
        OT = EP[1:count+1,]
        v = len(OT)
        if OT[v-1, 1] == 0:
            OT[v-1, 1] = dates[len(dates)-1]
    elif maxi == 0:
        OT = None
        print('No bubble or crisis periods found.')
    return OT