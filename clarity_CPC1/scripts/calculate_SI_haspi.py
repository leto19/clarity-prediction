import argparse
import csv
import json
import logging
import sys

import numpy as np
from clarity_core.config import CONFIG
from clarity_core.signal import find_delay_impulse, read_signal
from tqdm import tqdm
import transplant
from HASQI.HASPI import HASPI
sys.path.append("../projects/MBSTOI")
from MBSTOI import mbstoi
# module load MATLAB/2021a/binary
from scipy.signal import unit_impulse, resample
from soundfile import SoundFile
matlab = transplant.Matlab(jvm=False,desktop=False,print_to_stdout=False)
matlab.addpath("HASQI/matlab/")

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
    audiogram = np.delete(audiogram,4)
    audiogram = audiogram[:-1]
    return audiogram

def calculate_SI(
    scene,
    listener,
    system,
    clean_input_path,
    processed_input_path,
    input_listener_file,
    fs,
    gridcoarseness=1,
):
    """Run baseline speech intelligibility (SI) algorithm. MBSTOI
    requires time alignment of input signals. Here we correct for
    broadband delay introduced by the MSBG hearing loss model.
    Hearing aids also introduce a small delay, but this depends on
    the exact implementation. See projects/MBSTOI/README.md.

    Outputs can be found in text file sii.txt in /scenes folder.

    Args:
        scene (str): dictionary defining the scene to be generated
        listener (str): listener
        system (str): system
        clean_input_path (str): path to the clean speech input data
        processed_input_path (str): path to the processed input data
        fs (float): sampling rate
        gridcoarseness (int): MBSTOI EC search grid coarseness (default: 1)

    """
    logging.info(f"Running SI calculation: scene {scene}, listener {listener}")

    # Get non-reverberant clean signal
    clean = read_signal(f"{clean_input_path}/{scene}_target_anechoic.wav")

    # Get signal processed by HL and HA models
    """
    proc = read_signal(
        f"{processed_input_path}/{scene}_{listener}_{system}.wav",
    )
    """

    filename = f"{processed_input_path}/{scene}_{listener}_{system}.wav"
    #we  have to do this old school since read_signal only wants 44100Hz
    wave_file = SoundFile(filename)
    signal = wave_file.read()
    #resample raw HA outputs from 32000Hz to 44100Hz
    proc = resample(signal, int(CONFIG.fs * signal.shape[0] / wave_file.samplerate))
    
    # Calculate channel-specific unit impulse delay due to HL model and audiograms
    #Maybe don't need this but can't hurt (probably)
    ddf = read_signal(
        f"{processed_input_path}/{scene}_{listener}_{system}_HLddf-output.wav",
    )
    delay = find_delay_impulse(ddf, initial_value=int(CONFIG.fs / 2))

    if delay[0] != delay[1]:
        logging.info(f"Difference in delay of {delay[0] - delay[1]}.")

    maxdelay = int(np.max(delay))

    # Allow for value lower than 1000 samples in case of unimpaired hearing
    if maxdelay > 2000:
        logging.error(f"Error in delay calculation for signal time-alignment.")

    # Correct for delays by padding clean signals
    cleanpad = np.zeros((len(clean) + maxdelay, 2))
    procpad = np.zeros((len(clean) + maxdelay, 2))

    if len(procpad) < len(proc):
        raise ValueError(f"Padded processed signal is too short.")

    cleanpad[int(delay[0]) : int(len(clean) + int(delay[0])), 0] = clean[:, 0]
    cleanpad[int(delay[1]) : int(len(clean) + int(delay[1])), 1] = clean[:, 1]
    procpad[: len(proc)] = proc
    print(cleanpad.shape,procpad.shape)

    #get speaker audiogram info 
    with open(input_listener_file) as f:
        listener_dict = json.load(f) #probably not efficent to open this every time but w/e
    l_audiogram = format_audiogram(listener_dict[listener]['audiogram_levels_l'])
    r_audiogram = format_audiogram(listener_dict[listener]['audiogram_levels_r'])
    
    
    # Calculate intelligibility

    haspi_l = HASPI(cleanpad[:,0],44100,procpad[:,0],44100,l_audiogram,matlab)[0]
    haspi_r = HASPI(cleanpad[:,1],44100,procpad[:,1],44100,r_audiogram,matlab)[0]

    print(haspi_l,haspi_r)
    #input(">>>")
    """    sii = mbstoi(
        cleanpad[:, 0],
        cleanpad[:, 1],
        procpad[:, 0],
        procpad[:, 1],
        gridcoarseness=gridcoarseness,
    )
    """

    return [haspi_l,haspi_r]


def main(signals_filename, clean_input_path, processed_input_path,input_listener_file, output_sii_file, nsignals=None):
    """Main entry point, being passed command line arguments.

    Args:
        signals_filename (str): name of json file containing signal_metadata
        clean_input_path (str): path to clean input data
        processed_input_path (str): path to processed input data
        output_sii_file (str): name of output sii csv file
        nsignals (int, optional): Process first N signals. Defaults to None, implying all.
    """
    signals = json.load(open(signals_filename, "r"))

    f = open(output_sii_file, "a")
    writer = csv.writer(f)
    writer.writerow(["scene", "listener", "system", "HASPI"])

    # Process the first n signals if the nsignals parameter is set
    if nsignals and nsignals > 0:
        signals = signals[0:nsignals]

    for signal in tqdm(signals):
        listener = signal["listener"]
        scene = signal["scene"]
        system = signal["system"]
        sii = calculate_SI(
            scene,
            listener,
            system,
            clean_input_path,
            processed_input_path,
            input_listener_file,
            CONFIG.fs,
        )
        writer.writerow([scene, listener, system, sii])
        f.flush()

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsignals", type=int, default=None)
    parser.add_argument("signals_filename", help="json file containing signal_metadata")
    parser.add_argument("clean_input_path", help="path to clean input data")
    parser.add_argument("processed_input_path", help="path to processed input data")
    parser.add_argument("input_listener_file",help="file containing listener audiograms")
    parser.add_argument("output_sii_file", help="name of output sii csv file")
    args = parser.parse_args()

    main(
        args.signals_filename,
        args.clean_input_path,
        args.processed_input_path,
        args.input_listener_file,
        args.output_sii_file,
        args.nsignals,
    )
