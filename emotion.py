import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc,logfbank
import scipy.io.wavfile as wav
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn import svm
#import wavio

def read(file):
	return wav.read(file)

def preprocess(wav_rd):
	return wav_rd

def feature_extract(wav_rd):
	#(rate,sig) = wav_rd
	rate=wav_rd[0]
	sig=wav_rd[1]
	#print(rate)
	#print(sig)
	mfcc_feat = mfcc(sig,rate,nfft=1200)
	fbank_feat = logfbank(sig,rate,nfft=1200)
	return fbank_feat.flatten('F')[:8554]

def feature_select(X,y,k=400):
	#ANOVA-F
	fvalue_selector = SelectKBest(f_classif, k=400)

	X_kbest = fvalue_selector.fit_transform(X, y)
	outcome = fvalue_selector.get_support(indices=True)
	#print(outcome)
	#i=1/0

	print('Original number of features:', X.shape[1])
	print('Reduced number of features:', X_kbest.shape[1])
	return outcome,X_kbest

def train(X_train,y_train):
	classifier = svm.SVC(gamma='scale')
	classifier.fit(X_train, y_train)
	return classifier

def test(X_test,classifier):
	return classifier.predict(X_test)

if __name__=='__main__':

	#print(fbank_feat.shape)
	rd_file = [[] for _ in range(11)]
	X=np.zeros([11,8554])
	y=[1,2,3,1,2,3,1,2,3,1,3]

	rd_file[0] = read("Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav")
	rd_file[1] = read("Audio_Speech_Actors_01-24/Actor_01/03-01-02-01-01-01-01.wav")
	rd_file[2] = read("Audio_Speech_Actors_01-24/Actor_01/03-01-03-01-01-01-01.wav")
	rd_file[3] = read("Audio_Speech_Actors_01-24/Actor_02/03-01-01-01-01-01-02.wav")
	rd_file[4] = read("Audio_Speech_Actors_01-24/Actor_02/03-01-02-01-01-01-01.wav")
	rd_file[5] = read("Audio_Speech_Actors_01-24/Actor_02/03-01-03-01-01-01-02.wav")
	rd_file[6] = read("Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav")
	rd_file[7] = read("Audio_Speech_Actors_01-24/Actor_01/03-01-02-01-01-01-01.wav")
	rd_file[8] = read("Audio_Speech_Actors_01-24/Actor_01/03-01-03-01-01-01-01.wav")
	rd_file[9] = read("Audio_Speech_Actors_01-24/Actor_04/03-01-01-01-01-01-04.wav")
	#X[10] = feature_extract("Audio_Speech_Actors_01-24/Actor_04/03-02-02-01-01-01-04.wav").flatten('F')[:10998]
	rd_file[10] = read("Audio_Speech_Actors_01-24/Actor_01/03-01-03-01-01-01-01.wav")

	Z = feature_extract(preprocess(read("Audio_Speech_Actors_01-24/Actor_04/03-01-02-01-01-01-04.wav")))

	for i in range(len(rd_file)):
		rd_file[i]=preprocess(rd_file[i])
		X[i]=feature_extract(rd_file[i])

	#print(len(X))
	#print(type(X))
	#print(type(X[0]))

	(col,X_sel) = feature_select(X,y)
	print(col[:10])
	test_inst=Z[col].reshape(1,Z[col].shape[0])
	print(test_inst.shape)
	model = train(X_sel,y)
	#pred = model.predict(test_inst)
	pred = test(test_inst, model)
	print(pred)