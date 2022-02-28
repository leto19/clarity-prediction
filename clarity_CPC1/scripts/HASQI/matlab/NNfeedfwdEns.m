function model = NNfeedfwdEns(data,NNparam,Whid,Wout)
% Function to compute the neural network ensemble response to a set of
% inputs. The neural network is defined in NNfeedforwardZ.
%
% Calling arguments:
% data      array of features input to the neural network
% NNparam   vector of neural network paramters
% Whid      cell array of hidden layer weights for each network
% Wout      cell array of output layer weights for each network
%
% Returned value:
% model     neural network output vector averaged over the ensemble
%
% James M. Kates, 20 September 2011.

% Data and network parameters
ncond=size(data,1); %Number of conditions in the input data
K=length(Whid); %Number of networks in the ensemble

% Ensemble average of the predictions over the set of neural
% networks used for training
predict=zeros(ncond,K);
for k=1:K
    for n=1:ncond
        d=data(n,:);
        [~,output]=NNfeedforward(d,NNparam,Whid{k},Wout{k});
        predict(n,k)=output(2);
    end
end
model=mean(predict,2);

end

