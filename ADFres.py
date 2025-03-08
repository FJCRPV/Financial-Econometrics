import numpy as np

def ADFres(y, IC, adflag):
    T0 = len(y)
    T1 = len(y)-1
    const = np.ones(T1)
    dy = y[1:T0] - y[0:T1]
    x1 = const
    t = T1-adflag
    if IC > 0:
        ICC = np.full((adflag+1, 1), 0.0)
        betaM = np.full((adflag+1, 1), None)
        epsM = np.full((adflag+1, 1), None)
        for k in range(0, adflag+1):
            xx = x1[k:T1,None]
            dy01 = dy[k:T1,]
            if k > 0:
                x2 = np.column_stack((xx, np.full((T1-k, k) ,0)))
                for j in range(1, k+1):
                    x2[::, np.shape(xx)[1]+j-1] = dy[(k-j):(T1-j)]
                    
            else:
                x2 = xx
            betaM[k][0] = np.matmul(np.linalg.inv(np.matmul(x2.T, x2)), np.matmul(x2.T, dy01))
   

            epsM[k][0] = dy01-np.matmul(x2, betaM[k][0])
            npdf = sum(-1/2*np.log(2*np.pi)-1/2*(epsM[k][0]**2))

            if IC == 1:
                ICC[k,0] =- 2*npdf/t+2*np.shape(betaM[k][0])[0]/t
            elif IC == 2:
                ICC[k,0] =- 2*npdf/t+np.shape(betaM[k][0])[0]*np.log(t)/t
        lag0 = np.argmin(ICC)
        beta = betaM[lag0][0][0]
        eps = epsM[lag0][0]
        lag = lag0
    elif IC == 0:
        xx = x1[adflag:T1, None]
        dy01 = dy[adflag:T1,]
        if adflag > 0:
            x2 = np.column_stack((xx, np.full((t, adflag), 0)))
            for j in range(1, adflag+1):
                x2[::, np.shape(xx)[1]+j-1] = dy[(adflag-j):(T1-j)]
        else:
            x2 = xx
        #OLS regression
        beta = np.matmul(np.linalg.inv(np.matmul(x2.T, x2)), np.matmul(x2.T, dy01))
        eps = dy01-np.matmul(x2, beta)
        lag = adflag
    return {"beta":beta, "eps":eps, "lag":lag}