function [xcep,ycep] = ebm_CepCoef(xdB,ydB,thrCep,thrNerve,nbasis)
% Function to compute the cepstral correlation coefficients between the
% reference signal and the distorted signal log envelopes. The silence 
% portions of the signals are removed prior to the calculation based on the
% envelope of the reference signal. For each time sample, the log spectrum
% in dB SL is fitted with a set of half-cosine basis functions. The
% cepstral coefficients then form the input to the cepstral correlation
% calculation.
%
% Calling variables:
% xdB       subsampled reference signal envelope in dB SL in each band
% ydB	    subsampled distorted output signal envelope
% thrCep    threshold in dB SPL to include sample in calculation
% thrNerve  additive noise RMS for IHC firing (in dB)
% nbasis    number of cepstral basis functions to use
%
% Output:
% xcep      cepstral coefficient matrix for the ref signal (nsamp,nbasis)
% ycep      cepstral coefficient matrix for the output signal (nsamp,nbasis)
%             each column is a separate basis function, from low to high
%
% James M. Kates, 23 April 2015.
% Gammawarp version to fit the basis functions, 11 February 2019.
% Additive noise for IHC firing rates, 24 April 2019.

% Processing parameters
nbands=size(xdB,2); %Number of auditory frequency bands

% Mel cepstrum basis functions (mel cepstrum because of auditory bands)
freq=0:nbasis-1;
k=0:nbands-1;
cepm=zeros(nbands,nbasis);
for nb=1:nbasis
	basis=cos(freq(nb)*pi*k/(nbands-1)); %Basis functions
	cepm(:,nb)=basis/norm(basis);
end

% Find the reference segments that lie sufficiently above the quiescent rate
xLinear=10.^(xdB/20); %Convert envelope dB to linear (specific loudness)
xsum=sum(xLinear,2)/nbands; %Proportional to loudness in sones
xsum=20*log10(xsum); %Convert back to dB (loudness in phons)
index=find(xsum > thrCep); %Identify those segments above threshold
nsamp=length(index); %Number of segments above threshold

% Exit if not enough segments above zero
if nsamp <= 1
	xcep=[]; %Return empty matrices
    ycep=[];
	fprintf('Function ebm_CepCoef: Signal below threshold.\n');
	return;
end

% Remove the silent samples
xdB=xdB(index,:);
ydB=ydB(index,:);

% Add low-level noise to provide IHC firing jitter
xdB=ebm_AddNoise(xdB,thrNerve);
ydB=ebm_AddNoise(ydB,thrNerve);

% Compute the mel cepstrum coefficients using only those samples above
% threshold
xcep=xdB*cepm;
ycep=ydB*cepm;

% Remove the average value from the cepstral coefficients. The
% cepstral cross-correlation will thus be a cross-covariance, and there
% is no effect of the absolute signal level in dB.
for n=1:nbasis
    x=xcep(:,n);
    x=x - mean(x);
    xcep(:,n)=x;
    y=ycep(:,n);
    y=y - mean(y);
    ycep(:,n)=y;
end
