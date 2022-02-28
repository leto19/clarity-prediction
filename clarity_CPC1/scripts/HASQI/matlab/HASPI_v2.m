function [Intel,raw] = HASPI_v2(x,fx,y,fy,HL,Level1)
% Function to compute the HASPI intelligibility index using the 
% auditory model followed by computing the envelope cepstral
% correlation and BM vibration high-level covariance. The reference
% signal presentation level for NH listeners is assumed to be 65 dB
% SPL. The same model is used for both normal and impaired hearing. This
% version of HASPI uses a modulation filterbank followed by an ensemble of
% neural networks to compute the estimated intelligibility.
%
% Calling arguments:
% x			Clear input reference speech signal with no noise or distortion. 
%           If a hearing loss is specified, no amplification should be provided.
% fx        Sampling rate in Hz for signal x
% y			Output signal with noise, distortion, HA gain, and/or processing.
% fy        Sampling rate in Hz for signal y.
% HL		(1,6) vector of hearing loss at the 6 audiometric frequencies
%			  [250, 500, 1000, 2000, 4000, 6000] Hz.
% Level1    Optional input specifying level in dB SPL that corresponds to a
%           signal RMS = 1. Default is 65 dB SPL if argument not provided.
%
% Returned values:
% Intel     Intelligibility estimated by passing the cepstral coefficients
%           through a modulation filterbank followed by an ensemble of
%           neural networks.
% raw       vector of 10 cep corr modulation filterbank outputs, averaged
%           over basis funct 2-6.
%
% James M. Kates, 5 August 2013.

% Set the RMS reference level
if nargin < 6
    Level1=65;
end

% Auditory model for intelligibility
% Reference is no processing, normal hearing
itype=0; %Intelligibility model
[xenv,xBM,yenv,yBM,xSL,ySL,fsamp]=...
    eb_EarModel(x,fx,y,fy,HL,itype,Level1);

% ---------------------------------------
% Envelope modulation features
% LP filter and subsample the envelopes
fLP=320; %Envelope LP cutoff frequency, Hz
fsub=8*fLP; %Subsample to span 2 octaves above the cutoff frequency
[xLP,yLP]=ebm_EnvFilt(xenv,yenv,fLP,fsub,fsamp);

% Compute the cepstral coefficients as a function of subsampled time
nbasis=6; %Use 6 basis functions
thr=2.5; %Silence threshold in dB SL
dither=0.1; %Dither in dB RMS to add to envelope signals
[xcep,ycep]=ebm_CepCoef(xLP,yLP,thr,dither,nbasis);

% Cepstral coefficients filtered at each modulation rate
% Band center frequencies [2, 6, 10, 16, 25, 40, 64, 100, 160, 256] Hz
% Band edges [0, 4, 8, 12.5, 20.5, 30.5, 52.4, 78.1, 128, 200, 328] Hz
[xmod,ymod,cfmod]=ebm_ModFilt(xcep,ycep,fsub);

% Cross-correlations between the cepstral coefficients for the degraded and
% ref signals at each modulation rate, averaged over basis functions 2-6
% aveCM:  cep corr modulation filterbank outputs, ave over basis funct 2-6
aveCM=ebm_ModCorr(xmod,ymod);

% ---------------------------------------
% Intelligibility prediction
% Get the neural network parameters and the weights for an ensemble of
% 10 networks
[NNparam,Whid,Wout,b]=ebm_GetNeuralNet;

% Average the neural network outputs for the modulation filterbank values
model=NNfeedfwdEns(aveCM,NNparam,Whid,Wout);
model=model/b;

% Return the intelligibility estimate and raw modulation filter outputs
Intel=model;
raw=aveCM;
end
