function [xLP,yLP] = ebm_EnvFilt(xdB,ydB,fcut,fsub,fsamp)
% Function to lowpass filter and subsample the envelope in dB SL produced
% by the model of the auditory periphery. The LP filter uses a von Hann 
% raised cosine window to ensure that there are no negative envelope values
% produced by the filtering operation.
%
% Calling arguments
% xdB    matrix: env in dB SL for the ref signal in each auditory band
% ydB    matrix: env in dB SL for the degraded signal in each auditory band
% fcut   LP filter cutoff frequency for the filtered envelope, Hz
% fsub   subsampling frequency in Hz for the LP filtered envelopes
% fsamp  sampling rate in Hz for the signals xdB and ydB
%
% Returned values:
% xLP    LP filtered and subsampled reference signal envelope
%        Each frequency band is a separate column
% yLP    LP filtered and subsampled degraded signal envelope
%
% James M. Kates, 12 September 2019.

% Check the filter design parameters
xLP=[]; %Default error outputs
yLP=[];
if fsub > fsamp
    fprintf(' Error in ebm_EnvFilt: Subsampling rate too high.');
    return;
end
if fcut > 0.5*fsub
    fprintf(' Error in ebm_EnvFilt: LP cutoff frequency too high.');
    return;
end

% Check the data matrix orientation
% Require each frequency band to be a separate column
nrow=size(xdB,1); %Number of rows
ncol=size(xdB,2); %Number of columns
if ncol > nrow
    xdB=xdB';
    ydB=ydB';
end
nbands=size(xdB,2);
nsamp=size(xdB,1);

% Compute the lowpass filter length in samples to give -3 dB at fcut Hz
tfilt=1000*(1/fcut); %Filter length in ms
tfilt=0.7*tfilt; %Empirical adjustment of the filter length
nfilt=round(0.001*tfilt*fsamp); %Filter length in samples
nhalf=floor(nfilt/2);
nfilt=2*nhalf; %Force an even filter length

% Design the FIR LP filter using a von Hann window to ensure that there are
% no negative envelope values
benv=hanning(nfilt);
benv=benv/sum(benv);

% LP filter for the envelopes at fsamp
xenv=conv2(xdB,benv); %2-D convolution
xenv=xenv(nhalf+1:nhalf+nsamp,:); %Remove the filter transients
yenv=conv2(ydB,benv);
yenv=yenv(nhalf+1:nhalf+nsamp,:);

% Subsample the LP filtered envelopes
space=floor(fsamp/fsub);
index=1:space:nsamp;
xLP=xenv(index,:);
yLP=yenv(index,:);
end
