import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight
import itertools
from IPython.display import display
from utils import *

#get_ipython().magic('matplotlib inline')

if __name__ == '__main__':

	train_csv_file = 'audio_train.csv'
	test_csv_file = 'audio_test.csv'
	
	emotion_dict = {'neutral': 1, 'calm': 2, 'happy': 3, 'sad': 4, 'angry': 5, 'fearful': 6, 'disgust': 7, 'surprised': 8,}
	emo_keys = list(['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'])

	# emotion_dict = {'happy': 0, 'sad': 1, 'angry' : 2, 'surprised' : 3}
	# emo_keys = list(['happy', 'sad', 'angry', 'surprised'])
	raw_x_train, y_train, raw_x_test, y_test = read_raw_data_from_csv(train_csv_file, test_csv_file)
	#print(y_train[:10])
	print("Reading done")
	x_train, x_test = processed_data_from_raw(raw_x_train, raw_x_test)
	print("Processing done")
	# print(x_train)
	# print(y_train)
	# l=1/0

	# rf_classifier = RandomForestClassifier(n_estimators=2000, min_samples_split=45)
	# ml_model(rf_classifier, x_train, y_train, x_test, y_test, emotion_dict, emo_keys, model_name='rf_classifier')
	# print("Random Forest done")
	xgb_classifier = xgb.XGBClassifier(max_depth=7, learning_rate=0.008, objective='multi:softprob', 
                                   n_estimators=2000, sub_sample=0.5, num_class=len(emotion_dict), booster='gbtree', n_jobs=4)
	ml_model(xgb_classifier, x_train, y_train, x_test, y_test, emotion_dict, emo_keys, model_name='xgb_classifier')
	print("XGB done")
	# mlp_classifier = MLPClassifier(hidden_layer_sizes=(650, ), activation='relu', solver='adam',
 #                               batch_size='auto', learning_rate='adaptive', learning_rate_init=0.0001,
 #                               power_t=0.5, max_iter=1000, shuffle=True, random_state=None, tol=0.0001,
 #                               verbose=False, warm_start=False, momentum=0.8, nesterovs_momentum=True,
 #                               early_stopping=False, validation_fraction=0.05, beta_1=0.9, beta_2=0.999,
 #                               epsilon=1e-08)
	# ml_model(mlp_classifier, x_train, y_train, x_test, y_test, emotion_dict, emo_keys, model_name='mlp_classifier')
	# print("MLP done")
	# svc_classifier = LinearSVC(tol=0.0001, max_iter=1000)
	# ml_model(svc_classifier, x_train, y_train, x_test, y_test, emotion_dict, emo_keys, model_name='svc_classifier', svm=True)
	# print("SVC done")
	# mnb_classifier = MultinomialNB()
	# ml_model(mnb_classifier, x_train, y_train, x_test, y_test, emotion_dict, emo_keys, model_name='mnb_classifier')
	# print("MNB done")
	# lr_classifier = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
	# ml_model(lr_classifier, x_train, y_train, x_test, y_test, emotion_dict, emo_keys, model_name='lr_classifier')
	# print("LR done")

	# display_xgb_feature_importance(xgb_classifier)
	# xgb_evaluation_on_reduced_feature_set(x_train, y_train, x_test, y_test, xgb_classifier, emotion_dict, emo_keys)

	# perform_ensembles(y_test, [0,1,4], emotion_dict, emo_keys)

