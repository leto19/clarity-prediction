import transplant
import soundfile as sf
import numpy as np
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
#print(matlab.gpuDeviceTable)
#input("------")
start_time = timeit.default_timer()
haspi,raw = matlab.HASPI_v2((ref),(ref_fs),(deg),(deg_fs),(hearing_loss))
print("ref VS deg, w/ loss - HASPI:%s (ex time %ss)"%(haspi,(timeit.default_timer() - start_time)))

start_time = timeit.default_timer()
haspi,raw = matlab.HASPI_v2(ref,ref_fs,deg,deg_fs,no_loss)
print("ref VS deg, w/o loss - HASPI:%s (ex time %ss)"%(haspi,(timeit.default_timer() - start_time)))

haspi,raw = matlab.HASPI_v2(ref,ref_fs,ref,ref_fs,hearing_loss)
print("ref VS ref, w/ loss - HASPI:%s (ex time %ss)"%(haspi,(timeit.default_timer() - start_time)))

start_time = timeit.default_timer()
haspi,raw = matlab.HASPI_v2(ref,ref_fs,ref,ref_fs,no_loss)
print("ref VS ref, w/o loss - HASPI:%s (ex time %ss)"%(haspi,(timeit.default_timer() - start_time)))

