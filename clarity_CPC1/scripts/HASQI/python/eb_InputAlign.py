import numpy as np
def eb_InputAlign(x,y):
    nx = len(x)
    ny = len(y)
    nsamp = min(nx,ny)

    xy = np.cov(x[:nsamp],y[:nsamp])
    index = max(np.abs(xy))
    delay = nsamp - index
    
    fsamp=24000; #Cochlear model input sampling rate in Hz
    delay=delay - 2*fsamp/1000 #Back up 2 msec

    if delay >0:
        y = np.append(y[delay:ny],np.zeros(0,delay))
    else:
        delay = -delay
        y = np.append(np.zeros(0,delay),y[:ny-delay])

    xabs = np.abs(x)
    xmax = np.max(xabs)
    xthr = 0.001*xmax

    for n in range(0,nx-1):
        if xabs[n] > xthr: #First value above the threshold working forwards from the beginning
            nx0 = n
            break
    for n in range(nx,0,-1):
        if xabs[n] > xthr: #First value above the threshold working backwards from the end
            nx1 = n
            break
    if nx1 > ny:
        nx1 = ny
    xp = x[nx0:nx1]
    yp = y[nx0:nx1]
    return xp,yp

