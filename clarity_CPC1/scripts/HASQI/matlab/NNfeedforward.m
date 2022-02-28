function [hidden,output] = NNfeedforward(data,NNparam,Whid,Wout)
% Function to compute the outputs at each layer of a neural network given
% the input to the network and the weights. The activiation function is an
% offset logistic function that gives either a logsig or hyperbolic
% tangent; the outputs from each layer have been reduced by the offset. The
% structure of the network is an input layer, one hidden layer, and an
% output layer. The first values in vectors hidden and output are set to 1
% by the function, and the remaining values correspond to the outputs at
% each neuron in the layer.
%
% Calling arguments:
% data       feature vector input to the neural network
% NNparam    network parameters from NNinit
% Whid       matrix of weights for the hidden layer
% Wout       matrix of weights for the output layer
%
% Returned values:
% hidden     vector of outputs from the hidden layer
% output     vector of outputs from the output layer
%
% James M. Kates, 26 October 2010.

% Extract the parameters from the parameter array
nx=NNparam(1);
nhid=NNparam(2);
nout=NNparam(3);
beta=NNparam(4);
offset=NNparam(5);
again=NNparam(6);

% Initialize the array storage
x=zeros(nx+1,1);
hidden=zeros(nhid+1,1);
output=zeros(nout+1,1);

% Initialize the nodes used for constants
x(1)=1;
hidden(1)=1.0;
output(1)=1.0;

% Input layer
for i=2:nx+1
    x(i)=data(i-1);
end

% Response of the hidden layer
for j=2:nhid+1
    sumhid=sum(Whid(:,j-1).*x);
    hidden(j)=(again/(1.0 + exp(-beta*sumhid))) - offset;
end

% Response of the output layer
for k=2:nout+1
    sumout=sum(Wout(:,k-1).*hidden);
    output(k)=(again/(1.0 + exp(-beta*sumout))) - offset;
end

end
