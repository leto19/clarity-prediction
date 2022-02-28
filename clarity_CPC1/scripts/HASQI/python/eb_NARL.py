import numpy as np
from scipy.signal import firwin2
def eb_NARL(hearing_loss,nfir,fsamp):
    fmax = 0.5*fsamp

    aud = np.array([250,500,1000,2000,4000,6000])

    delay = np.zeroes((1,nfir+1))
    delay[nfir/2] = 1.0

    mloss = np.max(hearing_loss)
    if mloss >0:
        bias = np.array([-17,-8,1,-1,-2,-2])
        t3 = hearing_loss[1] + hearing_loss[2] + hearing_loss[3]
        if t3 >=180:
            xave = 0.05*t3
        else:
            xave = 9.0 + 0.116*(t3-180)

        gdB = xave + 0.31*hearing_loss + bias
        gdB = max(gdB,0)

        fv = np.array([0])
        fv = np.append(fv,aud)
        fv = np.append(fv,fmax)
        cfreq = np.arange(0,nfir)/nfir
        gain = np.interp(fmax*cfreq),fv,np.append(gdB[0],gdB,gdB[-1])
        glin = 10**(gain/20.0)
        nalr = firwin2(nfir,cfreq,glin,nyq=fmax)
    else:
        nalr = delay
    return nalr,delay
