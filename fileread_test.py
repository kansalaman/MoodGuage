import scipy.io.wavfile as wav
import glob
import wavio
for filename in glob.iglob("Audio_Speech_Actors_01-24"+"/*/*.wav",recursive=True):
    print(filename)
    print(wav.read(filename))
    print(wav.read(filename)[1].shape)
    print(wavio.read(filename))
    break