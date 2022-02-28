function [Xmod,Ymod,cf] = ebm_ModFilt(Xenv,Yenv,fsub)
% Function to apply an FIR modulation filterbank to the reference envelope
% signals contained in matrix Xenv and the processed signal envelope
% signals in matrix Yenv. Each column in Xenv and Yenv is a separate filter
% band or cepstral coefficient basis function. The modulation filters use a
% lowpass filter for the lowest modulation rate, and complex demodulation
% followed by a lowpass filter for the remaining bands. The onset and
% offset transients are removed from the FIR convolutions to temporally
% align the modulation filter outputs.
%
% Calling arguments:
% Xenv     matrix containing the subsampled reference envelope values. Each
%          column is a different frequency band or cepstral basis function
%          arranged from low to high.
% Yenv     matrix containing the subsampled processed envelope values
% fsub     envelope sub-sampling rate in Hz
%
% Returned values:
% Xmod     cell array containing the reference signal output of the
%          modulation filterbank. Xmod is of size {nchan,nmodfilt} where
%          nchan is the number of frequency channels or cepstral basis
%          functions in Xenv, and nmodfilt is the number of modulation
%          filters used in the analysis. Each cell contains a column vector
%          of length nsamp, where nsamp is the number of samples in each
%          envelope sequence contained in the columns of Xenv.
% Ymod     cell array containing the processed signal output of the
%          modulation filterbank.
% cf       vector of the modulation rate filter center frequencies
%
% James M. Kates, 14 February 2019.
% Two matrix version of gwarp_ModFiltWindow, 19 February 2019.

% Input signal properties
nsamp=size(Xenv,1);
nchan=size(Xenv,2);

% Modulation filter band cf and edges, 10 bands
% Band spacing uses the factor of 1.6 used by Dau
cf=[2, 6, 10, 16, 25, 40, 64, 100, 160, 256]; %Band center frequencies
nmod=length(cf);
edge=zeros(1,nmod+1); %Includes lowest and highest edges
edge(1:3)=[0, 4, 8]; %Uniform spacing for lowest two modulation filters
for k=4:nmod+1
%   Log spacing for remaining constant-Q modulation filters
    edge(k)=(cf(k-1)^2)/edge(k-1);
end

% Allowable filters based on envelope subsampling rate
fNyq=0.5*fsub;
index=edge < fNyq;
edge=edge(index); %Filter upper band edges less than Nyquist rate
nmod=length(edge) - 1;
cf=cf(1:nmod);

% Assign FIR filter lengths. Setting t0=0.2 gives a filter Q of about
% 1.25 to match Ewert et al. (2002), and t0=0.33 corresponds to Q=2 (Dau et
% al 1997a). Moritz et al. (2015) used t0=0.29. General relation Q=6.25*t0,
% compromise with t0=0.24 which gives Q=1.5
t0=0.24; %Filter length in sec for the lowest modulation frequency band
t=zeros(nmod,1);
t(1)=t0;
t(2)=t0;
t(3:nmod)=t0*cf(3)./cf(3:nmod); %Constant-Q filters above 10 Hz
nfir=2*floor(t*fsub/2); %Force even filter lengths in samples
nhalf=nfir/2;

% Design the family of lowpass windows
b=cell(nmod,1); %Filter coefficients, one filter impulse reponse per cell
for k=1:nmod
    b{k}=hann(nfir(k)+1); %Lowpass window
    b{k}=b{k}/sum(b{k}); %Normalize to 0-dB gain at cf
end

% Pre-compute the cosine and sine arrays
co=cell(nmod,1); %cosine array, one frequency per cell
si=cell(nmod,1); %sine array, one frequency per cell
n=1:nsamp;
for k=1:nmod
    if k == 1
        co{k}=1;
        si{k}=0;
    else
        co{k}=sqrt(2)*cos(pi*n*cf(k)/fNyq)';
        si{k}=sqrt(2)*sin(pi*n*cf(k)/fNyq)';    
    end
end

% Convolve the input and output envelopes with the modulation filters
Xmod=cell(nchan,nmod);
Ymod=Xmod;
for k=1:nmod %Loop over the modulation filters
    bk=b{k}; %Extract the lowpass filter impulse response
    nh=nhalf(k); %Transient duration for the filter
    c=co{k}; %Cosine and sine for complex modulation
    s=si{k};
    for m=1:nchan %Loop over the input signal vectors
%       Reference signal
        x=Xenv(:,m); %Extract the frequency or cepstral coefficient band
        u=conv((x.*c - 1i*x.*s),bk); %Complex demodulation, then LP filter
        u=u(nh+1:nh+nsamp); %Truncate the filter transients
        xfilt=real(u).*c - imag(u).*s; %Modulate back up to the carrier freq
        Xmod{m,k}=xfilt; %Save the filtered signal
        
%       Processed signal
        y=Yenv(:,m); %Extract the frequency or cepstral coefficient band
        v=conv((y.*c - 1i*y.*s),bk); %Complex demodulation, then LP filter
        v=v(nh+1:nh+nsamp); %Truncate the filter transients
        yfilt=real(v).*c - imag(v).*s; %Modulate back up to the carrier freq
        Ymod{m,k}=yfilt; %Save the filtered signal
    end
end
end
