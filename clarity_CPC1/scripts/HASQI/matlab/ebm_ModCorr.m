function aveCM = ebm_ModCorr(Xmod,Ymod)
% Function to compute the cross-correlations between the input signal
% time-frequency envelope and the distortion time-frequency envelope. The
% cepstral coefficients or envelopes in each frequency band have been
% passed through the modulation filterbank using function ebm_ModFilt.
%
% Calling variables:
% Xmod	   cell array containing the reference signal output of the
%          modulation filterbank. Xmod is of size {nchan,nmodfilt} where
%          nchan is the number of frequency channels or cepstral basis
%          functions in Xenv, and nmodfilt is the number of modulation
%          filters used in the analysis. Each cell contains a column vector
%          of length nsamp, where nsamp is the number of samples in each
%          envelope sequence contained in the columns of Xenv.
% Ymod	   subsampled distorted output signal envelope
% 
% Output:
% aveCM    modulation correlations averaged over basis functions 2-6
%          vector of size nmodfilt
%
% James M. Kates, 21 February 2019.

% Processing parameters
nchan=size(Xmod,1); %Number of basis functions
nmod=size(Xmod,2); %Number of modulation filters
small=1.0e-30; %Zero threshold

% Compute the cross-covariance matrix
CM=zeros(nchan,nmod);
for m=1:nmod
    for j=1:nchan
%       Index j gives the input reference band
        xj=Xmod{j,m}; %Input freq band j, modulation freq m
        xj=xj - mean(xj);
        xsum=sum(xj.^2);
%       Processed signal band 
        yj=Ymod{j,m}; %Processed freq band j, modulation freq m
        yj=yj - mean(yj);
        ysum=sum(yj.^2);
%       Cross-correlate the reference and processed signals
        if (xsum < small) || (ysum < small)
            CM(j,m)=0;
        else
            CM(j,m)=abs(sum(xj.*yj))/sqrt(xsum*ysum);
        end
    end
end
aveCM=mean(CM(2:6,:),1); %Average over basis functions 2 - 6
end
