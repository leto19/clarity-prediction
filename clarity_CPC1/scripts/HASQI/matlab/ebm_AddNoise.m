function zdB = ebm_AddNoise(ydB,thrdB)
% Function to add independent random Gaussian noise to the subsampled 
% signal envelope in each auditory frequency band.
%
% Calling arguments:
% ydB      subsampled envelope in dB re:auditory threshold
% thrdB    additive noise RMS level (in dB)
% Level1   an input having RMS=1 corresponds to Level1 dB SPL
%
% Returned values:
% zdB      envelope with threshold noise added, in dB re:auditory threshold
%
% James M. Kates, 23 April 2019.

% Additive noise sequence
rng('shuffle'); %Random number generator seed from clock
noise=thrdB*randn(size(ydB)); %Gaussian noise with RMS=1, then scaled

% Add the noise to the signal envelope
zdB=ydB + noise;
end
