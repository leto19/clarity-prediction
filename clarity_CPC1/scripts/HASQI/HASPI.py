
import numpy as np
def HASPI(ref,ref_fs,deg,deg_fs,hearing_loss,matlab):
    #print("ref",ref)
    #print("deg",deg)
    #print(hearing_loss)
    ref_fs = np.double(ref_fs)
    deg_fs = np.double(deg_fs)
    hearing_loss = hearing_loss.astype("float64")
    ref = ref.astype("float64")
    deg = deg.astype("float64")
    #print("-----")
    #print("ref after",ref,ref_fs)
    #print("deg after",deg,deg_fs)
    #print(hearing_loss)
    intel,raw = matlab.HASPI_v2(ref,ref_fs,deg,deg_fs,hearing_loss)
    #print(intel)
    return intel,raw


if __name__ == "__main__":
    import soundfile as sf
    import transplant
    import timeit
    ref,ref_fs = sf.read("guide/sig_clean.wav",dtype="float64")
    deg,deg_fs = sf.read("guide/sig_out.wav",dtype="float64")

    ref_fs = np.double(ref_fs)
    deg_fs = np.double(deg_fs)
    hearing_loss = np.array([10,10,20,30,40,55],dtype="float64")
    no_loss =  np.array([10,10,20,20,20,10],dtype="float64")
    print("loading matlab...")
    start_time = timeit.default_timer()

    matlab = transplant.Matlab(jvm=False,desktop=False,print_to_stdout=False)
    matlab.addpath("matlab/")
    end_time = timeit.default_timer()
    print("Done!(%s)"%(end_time - start_time))
    
    start_time = timeit.default_timer()
    score,_ = HASPI(ref,ref_fs,deg,deg_fs,hearing_loss,matlab)
    end_time = timeit.default_timer()
    print("HASPI: %s (%ss)"%(score,(end_time - start_time)))
