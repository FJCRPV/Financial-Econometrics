from ADFres import ADFres
import numpy as np
from PSY import PSY
import multiprocessing as mp

def cvPSYwmboot(y, swindow0="", Tb="", IC=0, adflag=0, nboot=199, useParallel=True):
    qe = [0.9,0.95,0.99]
    result = ADFres(y, IC, adflag)
    beta = result["beta"]
    eps = result["eps"]
    lag = result["lag"]

    T0 = len(eps)
    t = len(y)
    dy = y[1:t] - y[0:(t-1)]
    if type(beta) == np.float64:
        g = 1
    else:
        g = len(beta)




    if swindow0 == "":
        swindow0 = np.floor(t*(0.01+1.8/np.sqrt(t)))
    if Tb == "":
        print("Missing a value for 'Tb'")
        return False
    
    rN = np.random.random_integers(low=0, high=T0, size=(int(Tb), int(nboot)))
    wn = np.full((int(Tb), int(nboot)), np.random.normal(0,1))
    dyb = np.full((int(Tb)-1, int(nboot)), 0.0)


    dyb[0:lag+1] = dy[0:lag+1][0]
    for j in range(0,int(nboot)):
        if lag == 0:
            for i in range(lag, int(Tb)-1):
                dyb[i-1,j-1] = wn[i-lag-1,j-1]*eps[rN[i-lag-1,j-1]-1]
        elif lag > 0:
            x = np.full((int(Tb)-1, lag), 0.0)
            for i in range(lag, int(Tb)-1):
                x = np.full((int(Tb)-1, lag), 0.0)
                for k in range(0,lag):
                    x[i-1, k-1] = dyb[i-k-1, j-1]
                dyb[i-1, j-1] = np.matmul(x[i-1,], beta[1:g,0] + wn[i-lag-1,j-1]*eps[rN[i-lag-1, j-1]-1])

    yb0 = np.full((1, int(nboot)), y[0])
    dyb0 = np.append(yb0, dyb, axis=0)
    yb = np.cumsum(dyb0, axis=1)
    results = []
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(PSY, args=(yb[None,i].flatten(), swindow0, IC, adflag)) for i in range(1, int(nboot))]
    res = [results[i].get() for i in range(len(results))]
    pool.close()

    SPSY = np.asmatrix([np.max(res[i]) for i in range(0, len(res))])
    Q_PSY = np.asmatrix([np.quantile(SPSY,q) for q in np.array(qe).flatten()])
    return Q_PSY