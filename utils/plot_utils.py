import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import *

def plot_bar_char(x, y, title, y_label, font_scale=2):
	sns.set(style="darkgrid", font_scale=font_scale)
	ax = sns.barplot(x, y)
	ax.set_title(title)
	ax.set_ylabel(y_label)

	return plt


def plot_pie_chart(data, labels, colors, explode):
	plt.pie(data, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
	plt.axis('equal')

	return plt

def plot_explained_variance_vs_components(explained_variance_ratio):
	plt.figure(1, figsize=(8, 6))
	plt.axhline(y=1, color='r')
	plt.axhline(y=0.85, color='r')
	plt.plot(np.cumsum(explained_variance_ratio))
	plt.grid(True)
	plt.xlabel('number of components')
	plt.ylabel('cumulative explained variance')
	
def plot_confusion_matrix(y_test, y_pred, class_names):
	cnf_matrix = confusion_matrix(y_test, y_pred)
	df_cm = pd.DataFrame(
		cnf_matrix, index=class_names, columns=class_names, 
	)

	heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
	heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
	heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
	plt.ylabel('True label', fontsize=14)
	plt.xlabel('Predicted label', fontsize=14)
	plt.grid(True)
	plt.show()


def plot_roc_curve(y_test, y_pred_prob):
	fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
	plt.plot(fpr, tpr, label='AUC={:.4f}'.format(roc_auc_score(y_test, y_pred_prob)))
	plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
	         label='Luck', alpha=.8)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.title('ROC curve for diabetes readmission')
	plt.xlabel('False Positive Rate (1 - Specificity)')
	plt.ylabel('True Positive Rate (Sensitivity)')
	plt.grid(True)
	plt.legend(loc="lower right")
	plt.show()