import numpy as np
def eb_CenterFreq(nchan,shift=None):

    low_freq = float(80.0)
    high_freq = float(8000.0)
    
    ear_Q = 9.26449
    min_BW = 24.7

    if shift != None:
        k = 1 
        A = 165.49
        a = 2.1 #shift specified as a fraction of the total length

        #High Low location
        x_low = (1/a)*np.log10(k + (low_freq/A))
        x_high = (1/a)*np.log10(k + (high_freq/A))
        
        x_low = x_low*(1+shift)
        x_high = x_high*(1+shift)

        low_freq = A*(10^(a*x_low)-k)
        high_freq = A*(10^(a*x_high)-k)

    """
    % All of the following expressions are derived in Apple TR #35, "An
    % Efficient Implementation of the Patterson-Holdsworth Cochlear
    % Filter Bank" by Malcolm Slaney.
    """
    cf = -(ear_Q*min_BW)+np.exp((np.arange(1,nchan-1)[:, np.newaxis])*(-np.log(high_freq+ear_Q*min_BW) + \
        np.log(low_freq+ear_Q*min_BW))/(nchan-1))*(high_freq + ear_Q*min_BW)
    
    cf = np.insert(cf,0,high_freq)
    cf = np.append(cf,low_freq)
    cf = np.flipud(cf)
    return cf

if __name__ == '__main__':
    print(eb_CenterFreq(32))