import numpy as np
from scipy.interpolate import interp1d
def eb_LossParameters(hearing_loss,cfreq):


    aud = np.array([250,500,1000,2000,4000,6000])
    nfilt = len(cfreq) -1
    fv = np.array([cfreq[0]])
    fv = np.append(fv,aud)
    fv = np.append(fv,cfreq[nfilt])

    loss_temp = np.array(hearing_loss[0])
    loss_temp = np.append(loss_temp,hearing_loss)
    loss_temp = np.append(loss_temp,hearing_loss[5])

    loss = np.interp(cfreq,fv,loss_temp)
    loss = np.maximum(loss,0)
    CR = np.zeros((nfilt+1,1))
    for n in range(0,nfilt+1):
        CR[n] = float(1.25 + 2.25*(n-1)/(nfilt-1))
    #print(CR)

    maxOHC = 70*(1-(1./CR))
    thrOHC = 1.25*maxOHC
    
    attnOHC = np.zeros((nfilt+1,1))
    attnIHC = np.zeros((nfilt+1,1))
    for n in range(0,nfilt+1):
        if loss[n] < thrOHC[n]:
            attnOHC[n] = 0.8*loss[n]
            attnIHC[n] = 0.2*loss[n]
        else:
            attnOHC[n] = 0.8*thrOHC[n]
            attnIHC[n] = 0.2*thrOHC[n] + (loss[n] - thrOHC[n])

    BW = np.ones((nfilt+1,1))
    
    BW = BW + (attnOHC/50.0) + 2.0*(attnOHC/50.0)**6
    
    lowknee = attnOHC + 30;
    upamp = 30 + 70/CR
    CR = (100-lowknee)/(upamp+attnOHC - lowknee)
    return attnOHC,BW,lowknee,CR,attnIHC

if __name__ == '__main__':
    from eb_CenterFreq import eb_CenterFreq
    loss = [10, 10, 20, 30, 55, 55]
    cfreq = eb_CenterFreq(32)
    print(eb_LossParameters(loss,cfreq))