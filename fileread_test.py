import scipy.io.wavfile as wav
import glob
for filename in glob.iglob("Audio_Speech_Actors_01-24"+"/*/*.wav",recursive=True):
    # print(filename)
    try:
        if(wav.read(filename)[1].shape[1]):
            print(wav.read(filename)[1].shape)
    except:
        pass