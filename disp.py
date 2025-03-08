import pandas as pd

def disp (OT):
    if OT is None:
        print('No blubble or crisis periods found.')
    else:
        v = len(OT)
        dateStamps = pd.DataFrame()
        for j in range (0,v):
            if OT[j,0] == OT[j,1]:
                newEntry = pd.DataFrame([OT[j,0]], [OT[j,0]])
                dateStamps = dateStamps.append(newEntry, ignore_index = False)
            else:
                newEntry = pd.DataFrame([OT[j,1]], [OT[j,0]])
                dateStamps = dateStamps.append(newEntry, ignore_index = False)
        dateStamps.reset_index(inplace = True)
        dateStamps.rename(columns = {'index' : 'Start', 0 : 'End'}, inplace = True)
        print(dateStamps)