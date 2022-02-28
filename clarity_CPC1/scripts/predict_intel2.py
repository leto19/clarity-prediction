"""Predict final intelligibility scores from a set of MBSTOI scores.

Usage:
  python3 predict_intel.py <MBSTOI_CSV_FILE> <CPC1_TRAIN_JSON_FILE>
  <OUTPUT_CSV_FILE>

- <MBSTOI_CSV_FILE> - name of file containing the raw MBSTOI scores
- <CPC1_TRAIN_JSON_FILE> - the JSON file containing the CPC1 metadata
- <OUTPUT_CSV_FILE> - name of a csv file to which prediction will be written

Final stage of the baseline pipeline which maps MBSTOI scores 
(between 0 and 1) onto sentence intelligibility scores between (0 and 100).
The mapping is performed by first estimating the parameters of a sigmoid function
that minimise the RMS estimation error.

The script is provided as an example of the way in which the training
data should be treated when using it for development. The training data
is partitioned into training data and development evaluation data. i.e.
the sigmoid mapping is learnt from the training partition and applied to
the evaluation partition. A K-fold cross-validation set up is employed so
that, via repetition, a score can be computed for all responses in the 
training data set. 

Care needs to be taken when constructing the folds. In particular, the
training data has multiple responses originating from the same 'scene' -
either processed with a different hearing aid or processed for a different
listener. However, the final CPC1  evaluation dataset (released later) will contain a new set of previously unseen scenes. It is therefore important 
to evaluate during development in a scene-independent fashion. In this 
example script, scene-independence is maintained by splitting data 
on the 'scene' label field so that no signal originating from the same 
scene appears in more than one fold. 
"""


import argparse

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm
import soundfile as sf
from speechbrain.processing.features import STFT,ISTFT,spectral_magnitude
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from models.og_discrim import MetricDiscriminator


N_FOLDS = 5  # Number of folds to use in the n-fold cross validation

DATAROOT = "/fastdata/acp20glc/clarity_data/clarity_CPC1_data" 


def compute_feats(wavs,fs):
        """Feature computation pipeline"""
        wavs= torch.from_numpy(wavs)
        # T * C
        wavs = wavs.unsqueeze(0) #- change this if we start using real batches
        # B*T*C 
        stft = STFT(fs,n_fft= 1024,window_fn = torch.hamming_window)
        feats = stft(wavs)
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)
        return feats


def format_correctness(y):
    #convert correctness percentage to tensor
    y = torch.tensor([[y]])
    # normalize
    y = y/100
    return y


def test_model(model,test_data,optimizer,criterion):
    out_list = []
    model.eval()
    name_list = test_data["signal"]
    correctness_list = test_data["correctness"]
    #print(name_list)
    running_loss = 0.0
    loss_list = []
    for f_name,y in tqdm(zip(name_list,correctness_list)):
        audio,fs = sf.read("%s/clarity_data/HA_outputs/train/%s.wav"%(DATAROOT,f_name))
        #print(audio.shape)
        
        feats_l = compute_feats(audio[:,0],fs)
        feats_r = compute_feats(audio[:,1],fs)
        #print(feats_l.shape)
        #print(feats_r.shape)
        combined_feats = torch.cat(
                [feats_l, feats_r], 0
            ).unsqueeze(0)
        #print(combined_feats.shape)
        
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(combined_feats.float())
        
        # convert correctness value to tensor and normalize range 0 - 1
        y = format_correctness(y)
        
        print(outputs,y)
        loss = criterion(outputs, y)
        out_list.append(outputs.detach().numpy()[0]*100)
        loss_list.append(loss.item())
        # print statistics
        running_loss += loss.item()
    print("Average MSE loss: %s"%(sum(loss_list)/len(loss_list)))

    return out_list

def train_model(model,train_data,optimizer,criterion):
    name_list = train_data["signal"]
    correctness_list = train_data["correctness"]
    #print(name_list)

    running_loss = 0.0
    loss_list = []
    for f_name,y in zip(name_list,correctness_list):
        #Load the stero audio file 
        audio,fs = sf.read("%s/clarity_data/HA_outputs/train/%s.wav"%(DATAROOT,f_name))
        
        # compute the feats for each stero channel 
        feats_l = compute_feats(audio[:,0],fs)
        feats_r = compute_feats(audio[:,1],fs)
        
        # combine the channel level features 
        combined_feats = torch.cat(
                [feats_l, feats_r], 0
            ).unsqueeze(0)
        
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(combined_feats.float())
        
        # convert correctness value to tensor and normalize range 0 - 1
        y = format_correctness(y)
        
        print(outputs,y)
        loss = criterion(outputs, y)
        
        loss.backward()
        optimizer.step()
        loss_list.append(loss)
        running_loss += loss.item()
        print("Average loss: %s"%(sum(loss_list)/len(loss_list)))
        
    return model,optimizer,criterion



    



def main(mbstoi_file_csv, intel_file_json, prediction_file_csv):

    # Load the mbstoi data and the intelligibility data
    df_mbstoi = pd.read_csv(mbstoi_file_csv)
    df_intel = pd.read_json(intel_file_json)

    #print(df_intel['signal'])
    # Merge into a common dataframe
    data = pd.merge(
        df_mbstoi,
        df_intel[["scene", "listener", "system", "correctness","signal"]],
        how="left",
        on=["scene", "listener", "system"],
    )
    data["predicted"] = np.nan  # Add column to store intel predictions
    

    # use this line for testing :) 
    #data = data[:100]
    
    # Make a unique list of all the scenes appearing in the data
    scenes = data.scene.unique()

    #set up the torch objects
    model = MetricDiscriminator()
    criterion = nn.MSELoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=0.5)
    optimizer = optim.Adam(model.parameters(),lr=0.01)
    for fold in tqdm(range(N_FOLDS)):
        # Split the train scenes and the test scenes
        test_scenes = set(scenes[fold : len(scenes) : N_FOLDS])
        train_scenes = set(scenes) - set(test_scenes)


        # Using only the data corresponding to the train set scenes
        train_data = data[data.scene.isin(train_scenes)].sample(frac = 1)
        model,optimizer,criterion = train_model(model,train_data,optimizer,criterion)


        test_data = data[data.scene.isin(test_scenes)]
        #get predictions for the test set 
        predictions = test_model(model,test_data,optimizer,criterion)
        #print(predictions)

        # Applying them only to the test set scenes
        data.loc[data.scene.isin(test_scenes), ["predicted"]] = predictions
      
    
    # There should be no scenes without a prediction
    assert data["predicted"].isna().sum() == 0

    # Save data as csv
    data[["scene", "listener", "system", "predicted"]].to_csv(prediction_file_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mbstoi_csv_file", help="csv file containing MBSTOI predictions")
    parser.add_argument(
        "cpc1_train_json_file", help="JSON file containing the CPC1 training metadata"
    )
    parser.add_argument(
        "out_csv_file", help="output csv file containing the intelligibility predictions"
    )
    args = parser.parse_args()
    main(args.mbstoi_csv_file, args.cpc1_train_json_file, args.out_csv_file)
