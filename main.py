'''
PSY Monitor Python Package
Authors: Francisco Perestrello, Jorge Gouveia, and Sami Benazzouz
'''


#Usage
#For the illustration purposes we will use data on the credit risk in the European sovereign sector, that is proxied by an index constructed as a GDP weighted 10-year government bond yield of the GIIPS (Greece, Ireland, Italy, Portugal, and Spain) countries.

#Let's walk through some bascis. First load the necessary libraries.

import pyreadr
import numpy as np

from cvPSYwmboot import cvPSYwmboot
from PSY import PSY

from locate import locate
from disp import disp


if __name__ == "__main__":
    
    #Next, get data on GIIPS spread and define a few parameters for the test and the simulation
    spread = pyreadr.read_r('spread.rda')["spread"]
    snp = pyreadr.read_r('snp.rda')["snp"]
    y = spread["value"].values
    obs = len(y)
    swindow0 = np.floor(obs*(0.01+1.8/np.sqrt(obs))) # set minimal window size
    IC = 2 # use BIC to select the number of lags
    adflag = 6 # set the maximum number of lags to 6
    yr = 2
    Tb = 12*yr + swindow0 - 1 # set the control sample size
    nboot = 99 # set the number of replications for the bootstrap
    
    #Then, estimate the PSY test statistic using PSY() and its corresponding bootstrap-based critical values using cvPSYwmboot().
    bsadf = PSY(y, swindow0, IC, adflag)
    quantilesBsadf = cvPSYwmboot(y, swindow0, Tb, IC, adflag, Tb, nboot)
    
    #Now identify crisis periods, defined as periods where the test statistic is above its corresponding critical value, using the locate() function.
    dim = obs - swindow0 + 1
    monitorDates = spread["date"].values[int(swindow0)-1:obs]
    quantiles95 = np.matmul(quantilesBsadf.T,np.full((1, int(dim)), 1.0))
    ind95 = (bsadf>quantiles95[1,])*1
    ind95 = np.asarray(ind95)[0]
    periods = locate(ind95, monitorDates) # locate crisis periods
    
    #Finally, print a table that holds the identified crisis periods with the help of the disp() function.
    disp(periods) # generate table that holds crisis periods