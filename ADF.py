import numpy as np 

def ADF(y, IC=0, adflag=0):
    if IC!=0 and IC!=1 and IC!=2:
        return "ERROR : param IC  An integer. 0 for fixed lag (default) order 1 for AIC and 2 for BIC   (default = 0)."
    T0 = len(y)
    T1 = len(y) - 1
    const = np.ones(T1)
    y1 = y[0:T1]
    y0 = y[1:T0]
    dy = y0 - y1
    x1 = np.column_stack((np.transpose(y1), np.transpose(const)))
    t = T1 - adflag
    dof = t - adflag - 2
    ICC = np.full((adflag + 1, 1), 0.0)
    ADF = np.full((adflag + 1, 1), 0.0)
    if IC>0:
        for k in range(0, adflag+1): # change k to start from 0
            xx = x1[k:T1]

            dy01 = dy[k:T1]

            if k > 0:
                x2 = np.column_stack((xx, np.full((T1-k, k), 0)))

                for j in range(1, k+1):
                    x2[::, np.shape(xx)[1]+j-1] = dy[(k-j):(T1-j)]
            else:
                x2 = xx


            beta = np.matmul(np.linalg.inv(np.matmul(x2.T, x2)), np.matmul(x2.T, dy01))

            eps = dy01 - np.matmul(x2, beta)
            npdf = sum(-1/2*np.log(2*np.pi)-1/2*(eps**2))

            if IC == 1:
                #AIC
                ICC[k,0] =- 2*npdf/t+2*np.shape(beta)[0]/t
            elif IC == 2:
                #BIC
                ICC[k,0] =- 2*npdf/t+np.shape(beta)[0]*np.log(t)/t
            se = np.matmul(eps.T, eps/dof)

            sig = np.sqrt(np.diagonal(np.full((np.shape(beta)[0], np.shape(beta)[0]), se)*np.linalg.inv(np.matmul(x2.T, x2))))

            ADF[k,0] = beta[0]/sig[0]

            lag0 = np.argmin(ICC)

            ADFlag = ADF[lag0][0]

    elif IC == 0:
        xx = x1[adflag:T1,]
        dy01 = dy[adflag:T1]

        if adflag > 0:
            x2 = np.column_stack((xx, np.full((t, adflag) ,0)))
            for j in range(1, adflag+1):
                x2[::, np.shape(xx)[1]+j-1] = dy[adflag-j:T1-j]
        else:
            x2 = xx

        #OLS regression
        beta = beta = np.matmul(np.linalg.inv(np.matmul(x2.T, x2)), np.matmul(x2.T, dy01))
        eps = dy01-np.matmul(x2, beta)
        se = np.matmul(eps.T,eps/dof)
        sig = np.sqrt(np.diagonal(np.full((np.shape(beta)[0], np.shape(beta)[0]), se)*np.linalg.inv(np.matmul(x2.T, x2))))
        ADFlag = beta[0]/sig[0]


    return ADFlag