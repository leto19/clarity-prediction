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
from GHA import audiogram

import numpy as np
import numpy
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm
import soundfile as sf
from speechbrain.processing.features import STFT,ISTFT,spectral_magnitude
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import transplant
from models.og_discrim import MetricDiscriminator_no_ref
import speechbrain as sb
from torch.utils.data import DataLoader
from HASQI.HASPI import HASPI
from scipy.signal import unit_impulse, resample
N_FOLDS = 5  # Number of folds to use in the n-fold cross validation

DATAROOT = "/fastdata/acp20glc/clarity_data/clarity_CPC1_data" 
df_listener = pd.read_json("metadata/listeners.CPC1_train.json")
#matlab = transplant.Matlab(jvm=False,desktop=False,print_to_stdout=False)
#matlab.addpath("HASQI/matlab/")
def compute_feats(wavs,fs):
        """Feature computation pipeline"""
        #wavs= torch.from_numpy(wavs)
        
        #wavs = wavs.unsqueeze(0) #- change this if we start using real batches
        # B*T*C 
        #print(wavs.shape)
        # B*C*T
        #wavs = wavs.transpose(2,1)
        #print(wavs.shape)
        
        stft = STFT(fs,n_fft= 1024,window_fn = torch.hamming_window)
        feats = stft(wavs)
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)
        print(feats.shape)
        return feats


from speechbrain.processing.speech_augmentation import Resample
resampler = Resample(orig_freq=32000, new_freq=44100,lowpass_filter_width=4)

def audio_pipeline(path,fs=32000):
    audio = sb.dataio.dataio.read_audio_multichannel(path)
    #print("resampling...")
    #audio = resampler(audio)
    #print(audio.shape)
    audio = resample(audio, int(44100 * audio.shape[0] / 32000))
    audio = np.mean(audio,axis = 1)
    print(audio.shape)

    #sf.write("test_out.wav",audio,44100)
    #input(">>")
    #audio = audio.transpose(1,0)
    
    #print(audio.shape)
    return audio 


def format_correctness(y):
    #convert correctness percentage to tensor
    y = torch.tensor([y])
    # normalize
    y = y/100
    return y


def get_mean(scores):
    out_list = []
    for el in scores:
        el = el.strip("[").strip("]").split(",")
        el = [float(a) for a in el]
        #print(el,type(el))
        out_list.append(sum(el)/len(el))
    return torch.Tensor(out_list).unsqueeze(0)

def test_model(model,test_data,optimizer,criterion):
    out_list = []
    model.eval()
    name_list = test_data["signal"]
    correctness_list = test_data["correctness"]
    #print(name_list)
    running_loss = 0.0
    loss_list = []
    for f_name,y in tqdm(zip(name_list,correctness_list)):
        audio= audio_pipeline("%s/clarity_data/HA_outputs/train/%s.wav"%(DATAROOT,f_name))
        print(audio.shape,type(audio))
        audio = torch.from_numpy(audio).unsqueeze(0)
        print(audio.shape,type(audio))

        combined_feats = compute_feats(audio,32000)
        combined_feats = combined_feats.transpose(3,1) # B* T * C

        print(combined_feats.shape)

        print(combined_feats.shape)
        # forward + backward + optimize
        outputs = model(combined_feats.float())
        
        # convert correctness value to tensor and normalize range 0 - 1
        y = format_correctness(y)
        
        print(outputs,y)
        loss = criterion(outputs, y)
        print(outputs)
        print(outputs.detach().numpy()[0][0]*100)
        #input(">>")
        out_list.append(outputs.detach().numpy()[0][0]*100)

        loss_list.append(loss.item())
        # print statistics
        running_loss += loss.item()
    print("Average MSE loss: %s"%(sum(loss_list)/len(loss_list)))

    return out_list



def batch_HASPI(clean_batch,pred_batch,audiogram_batch):
    haspi_list = []
    for ref,pred,aud in zip(clean_batch,pred_batch,audiogram_batch):
        ref_l = ref[0,:].numpy()
        ref_r = ref[1,:].numpy()
        deg_l = pred[0,:].numpy()
        deg_r = pred[1,:].numpy()
        fs = 44100
        aud_l = aud[0].numpy()
        sf.write("ref_l.wav",ref_l,fs)
        sf.write("deg_l.wav",deg_l,fs)
        input(">>>>>>>>")
        aud_l = format_audiogram(aud_l)
        aud_r = aud[1].numpy()
        aud_r = format_audiogram(aud_r)
        haspi_l = HASPI(ref_l,fs,deg_l,fs,aud_l,matlab)[0]
        haspi_r = HASPI(ref_r,fs,deg_r,fs,aud_r,matlab)[0]
        print(haspi_l,haspi_r)
        mean_haspi = (haspi_l + haspi_r) /2
        haspi_list.append(torch.Tensor([mean_haspi]))
    return haspi_list #tensor of scores


def format_audiogram(audiogram):
    """
     "audiogram_cfs": [
      250,
      500,
      1000,
      2000,
      3000,
      4000,
      6000,
      8000
    ],
    
    TO
    [250, 500, 1000, 2000, 4000, 6000
    """
    audiogram = numpy.delete(audiogram,4)
    audiogram = audiogram[:-1]
    return audiogram
    



def train_model(model,train_data,optimizer,criterion):
    name_list = train_data["signal"]
    correctness_list = train_data["correctness"]
    haspi_list = train_data["HASPI"]
    scene_list = train_data["scene"]
    listener_list = train_data["listener"]
    #print(name_list)
    #columns_titles = ["signal",'scene', 'listener', 'system', 'mbstoi', 'correctness', 'predicted']
    #train_data = train_data.reindex(columns_titles)
    #train_data = train_data.to_dict()
    running_loss = 0.0
    loss_list = []
   
    train_dict = {}
    for name,corr,scene,lis,haspi in  zip(name_list,correctness_list,scene_list,listener_list,haspi_list):
        train_dict[name] = {"signal": name,"correctness":corr,"scene": scene,"listener":lis,"haspi":haspi}
        #print(train_dict[name])
    #print(train_dict)
    #print(train_dict)
    dynamic_items = [
        {"func": lambda l: format_correctness(l),
        "takes": "correctness",
        "provides": "formatted_correctness"},
        {"func": lambda l: audio_pipeline("%s/clarity_data/HA_outputs/train/%s.wav"%(DATAROOT,l),32000),
        "takes": "signal",
        "provides": "wav"},
        #{"func": lambda l: audio_pipeline("%s/clarity_data/scenes//%s_target_anechoic.wav"%(DATAROOT,l),44100),
        #"takes": "scene",
        #"provides": "clean_wav"},
        #{"func": lambda l: convert_audiogram(l),
        #"takes": "listener",
        #"provides": "audiogram_np"}
    ]
    train_set = sb.dataio.dataset.DynamicItemDataset(train_dict,dynamic_items)
    #train_set.set_output_keys(["wav","clean_wav", "formatted_correctness","audiogram_np","haspi"])
    train_set.set_output_keys(["wav", "formatted_correctness","haspi"])

    my_dataloader = DataLoader(train_set,10,collate_fn=sb.dataio.batch.PaddedBatch)
    print("starting training...")
    for batch in tqdm(my_dataloader):
        wavs,correctness,haspi = batch
        
        #print(audiogram[0])
        correctness =correctness.data
        #print(wavs_clean.data.shape)
        #print(wavs.data.shape)

        scores = get_mean(haspi).T
        #print(scores)
        #wavs.data = resampler(wavs.data)
        feats = compute_feats(wavs.data,32000) #B *  C * T
        feats = feats.unsqueeze(1)
        #feats = feats.transpose(3,1) # B* T * C
        #feats_clean = compute_feats(wavs_clean.data,44100)
        #feats_clean.transpose(3,1)
        #print(feats_clean.shape)
       
        #print(feats.shape)
        
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(feats.float())
        #print(outputs)
        # convert correctness value to tensor and normalize range 0 - 1
        #print(outputs.shape,correctness.shape)
        #print(outputs,correctness)
        """
        for x,y in zip(outputs.detach().numpy(),correctness.detach().numpy()):
            print("P: %s | T: %s"%(x,y))
        loss = criterion(outputs, correctness)
        """
        for x,y in zip(outputs.detach().numpy(),scores.detach().numpy()):
            print("P: %s | T: %s"%(x,y))
        loss = criterion(outputs, scores)
        loss.backward()
        optimizer.step()
        loss_list.append(loss)
        running_loss += loss.item()
        print("Average loss: %s"%(sum(loss_list)/len(loss_list)))
    
    return model,optimizer,criterion


def convert_audiogram(listener):
    audiogram_l =  np.array(df_listener[listener]["audiogram_levels_l"])
    audiogram_r =  np.array(df_listener[listener]["audiogram_levels_r"])
    audiogram = [audiogram_l,audiogram_r]
    return audiogram

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
    #data = data[:50]
    
    # Make a unique list of all the scenes appearing in the data
    scenes = data.scene.unique()

    #set up the torch objects
    model = MetricDiscriminator_no_ref()
    criterion = nn.MSELoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=0.5)
    optimizer = optim.Adam(model.parameters(),lr=0.1)
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
    predictions = test_model(model,data,optimizer,criterion)
    test_scenes = data["scene"]
    data.loc[data.scene.isin(test_scenes), ["predicted"]] = predictions

    print(data["predicted"])
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
