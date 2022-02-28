from eb_CenterFreq import eb_CenterFreq
from eb_LossParameters import eb_LossParameters
from eb_Resample24kHz import eb_Resample24kHz
from eb_InputAlign import eb_InputAlign
from eb_NARL import eb_NARL
import numpy as np
def eb_EarModel(x,x_fs,y,y_fs,hearing_loss,itype,level1):


    nchan = 32
    mdelay = 1
    cfreq = eb_CenterFreq(nchan)

    attnOHCy,BWminy,lowkneey,CRy,attnIHCy = eb_LossParameters(hearing_loss,cfreq)


    hearing_loss_max = [100,100,100,100,100,100]
    shift = 0.02
    cfreq1 = eb_CenterFreq(nchan,shift)
    _,BW1,_,_,_ = eb_LossParameters(hearing_loss_max,cfreq1)

    x = x[:].T
    y = y[:].T.reshape

    x24,_ = eb_Resample24kHz(x,x_fs)
    y24,fsamp = eb_Resample24kHz(y,y_fs)

    nxy = min(len(x24),len(y24))
    x24 = x24[:nxy]
    y24 = y24[:nxy]

    x24,y24 = eb_InputAlign(x24,y24)
    nsamp = len(x24)

    if itype == 1:
        nfir = 140
        nalr,_ = eb_NARL(hearing_loss,nfir,fsamp)
        x24 = np.convolve(x24,nalr)
        x24 = x24[nfir:nfir+nsamp]

    xmid = eb_MiddleEar