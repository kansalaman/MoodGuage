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

#get_ipython().magic('matplotlib inline')

def read_raw_data_from_csv(train_csv_file, test_csv_file):
	x_train = pd.read_csv(train_csv_file)
	x_test = pd.read_csv(test_csv_file)
	y_train = x_train['label']
	y_test = x_test['label']
	return x_train, y_train, x_test, y_test


def processed_data_from_raw(x_train, x_test):
	#print(x_train.shape, x_test.shape)
	#print("Doing1")
	cl_weight = dict(pd.Series(x_train['label']).value_counts(normalize=True))
	#print("Done1")
	#print(dict(pd.Series(x_train['label']).value_counts()))

	del x_train['label']
	#print("Done2")
	del x_test['label']
	#print("Done3")
	del x_train['file_name']
	del x_test['file_name']
	del x_train['Unnamed: 0']
	del x_test['Unnamed: 0']
	return x_train, x_test

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
    
def one_hot_encoder(true_labels, num_records, num_classes):
	#print(true_labels)
	temp = np.array(true_labels[:num_records])
	true_labels = np.zeros((num_records, num_classes))
	true_labels[np.arange(num_records), temp] = 1
	#print(true_labels)
	#print("-----")
	return true_labels

def show_evaluation_metrics(y_test, pred_probs, emotion_dict, emo_keys, cm=True):
	pred = np.argmax(pred_probs, axis=-1)
	#print(pred)
	'''for i in range(50):
		print(y_test[i], pred[i])
	print("-----")'''
	one_hot_true = one_hot_encoder(y_test, len(pred), len(emotion_dict))
	print('Test Set Accuracy =  {0:.3f}'.format(accuracy_score(y_test, pred)))
	print('Test Set F-score =  {0:.3f}'.format(f1_score(y_test, pred, average='macro')))
	print('Test Set Precision =  {0:.3f}'.format(precision_score(y_test, pred, average='macro')))
	print('Test Set Recall =  {0:.3f}'.format(recall_score(y_test, pred, average='macro')))
	print(confusion_matrix(y_test, pred))
	#if cm:
	#	plot_confusion_matrix(confusion_matrix(y_test, pred), classes=emo_keys)

def ml_model(classifier, x_train, y_train, x_test, y_test, emotion_dict, emo_keys, model_name, svm=False, dump=True):
	#print(x_train.shape)
	#print(y_train[:20])
	classifier.fit(x_train, y_train)
	if svm:
		#print("Here")
		predictions = classifier.predict(x_test)
		predictions = one_hot_encoder(predictions, len(predictions), len(emotion_dict))
	else:
		#print('here I am')	
		predictions = classifier.predict_proba(x_test)
		#print(predictions)
	#print(x_test.shape)
	#print(predictions.shape)
	show_evaluation_metrics(y_test, predictions, emotion_dict, emo_keys)
	if dump:
		with open('pred_probas/{}.pkl'.format(model_name), 'wb') as f:
			pickle.dump(predictions, f)

def display_xgb_feature_importance(xgb_classifier):
	ax = xgb.plot_importance(xgb_classifier, max_num_features=10, height=0.5, show_values=False)
	fig = ax.figure
	fig.set_size_inches(8, 8)

def xgb_evaluation_on_reduced_feature_set(x_train, y_train, x_test, y_test, xgb_classifier, emotion_dict, emo_keys):
	contribution_scores = xgb_classifier.feature_importances_
	top_n = [3, 5, 6, 7, 8]
	for n in top_n:
		threshold = np.argsort(contribution_scores)[::-1][:n][-1]
		print('Stats for top {} features:'.format(n))
		selection = SelectFromModel(xgb_classifier, threshold=contribution_scores[threshold], prefit=True)
		select_x_train = selection.transform(x_train)
		select_x_test = selection.transform(x_test)
		select_xgb_classifier = xgb.XGBClassifier(max_depth=7, learning_rate=0.008, objective='multi:softprob', 
                                              n_estimators=1200, sub_sample = 0.8, num_class = len(emotion_dict),
                                              booster='gbtree', n_jobs=4)
		ml_model(select_xgb_classifier, select_x_train, y_train, select_x_test, y_test, emotion_dict, emo_keys, model_name='garbage', dump=False)
		'''select_xgb_classifier.fit(select_x_train, y_train)
		predictions = select_xgb_classifier.predict_proba(select_x_test)
		show_evaluation_metrics(y_test, predictions, cm = False)'''

def perform_ensembles(y_test, models, emotion_dict, emo_keys):
	x = [0 for _ in range(7)]
	ensemble_pred_probs = 0.0

	with open('pred_probas/rf_classifier.pkl', 'rb') as f:
		x[0] = pickle.load(f)

	with open('pred_probas/xgb_classifier.pkl', 'rb') as f:
		x[1] = pickle.load(f)

	with open('pred_probas/svc_classifier.pkl', 'rb') as f:
		x[2] = pickle.load(f)

	with open('pred_probas/mnb_classifier.pkl', 'rb') as f:
		x[3] = mnb_pred_probs = pickle.load(f)

	with open('pred_probas/mlp_classifier.pkl', 'rb') as f:
		x[4] = mlp_pred_probs = pickle.load(f)

	with open('pred_probas/lr_classifier.pkl', 'rb') as f:
		x[5] = pickle.load(f)

	'''with open('pred_probas/lstm_classifier.pkl', 'rb') as f:
		x[6] = pickle.load(f)'''

	# Average of the predicted probabilites
	for i in models:
		ensemble_pred_probs+=x[i]
	ensemble_pred_probs/=len(models)
	# Show metrics
	show_evaluation_metrics(y_test, ensemble_pred_probs, emotion_dict, emo_keys)

def up_sample(df,label):
    emo_df = df[df['label']==label]
    for i in range(10):
        df = df.append(emo_df)
    return df
