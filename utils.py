import pickle
import pandas as pd 

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

def plot_trial_results(trials):
  exp_list =[]
  for trial in trials.trials:
    exp = {'id': trial['tid'] , 'accuracy':-trial['result']['loss']}
    exp_list.append(exp)
  __df= pd.DataFrame(exp_list)
  __df.plot(x='id', y='accuracy', kind='scatter', title='Accuracy over Trials')
  return None

def df_from_trial(trials):
  exp_list =[]
  for trial in trials.trials:
    exp = {'id': trial['tid'] , 'vals':trial['misc']['vals'], 'time': trial['result']['time'] ,'accuracy':-trial['result']['loss'], }
    exp_list.append(exp)
  aux = pd.DataFrame(exp_list)
  vals_df = pd.DataFrame(list(aux.vals))
  final_df = pd.concat([aux.drop('vals', axis=1), vals_df], axis=1)
  return final_df

def plot_time(df):
  df[['time', 'accuracy']].sort_values('time').plot(x='time', y='accuracy',kind='scatter', title='Accuracy vs Time of each Trial')
  return None

def load_trial(path):
  with open(path, 'rb') as f:
    trials=pickle.load(f)
  return trials

def compute_results(y_true, y_pred ):
  accuracy = accuracy_score(y_true, y_pred)
  f1_macro = f1_score(y_true, y_pred, average='macro')
  recall_macro = recall_score(y_true, y_pred, average='macro')
  precision_macro = precision_score(y_true, y_pred, average='macro')
  return {'accuracy': accuracy,
          'f1_macro': f1_macro,
          'recall_macro': recall_macro,
          'precision_macro': precision_macro}

import matplotlib.pyplot as plt
plt.style.use('ggplot')

#função para visualizar o historico de treinamento da rede
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()



def save_figs(path,learner,name):
  fig_path =  path + str(name)
  learner.plot(plot_type = 'loss', return_fig = True).savefig(fig_path+'/' + 'loss')
  learner.plot(plot_type = 'accuracy', return_fig = True).savefig(fig_path+'/' + 'acc')

def get_gini_importance(x_train, y_train):
  model = DecisionTreeClassifier()
  vectorizer = TfidfVectorizer(ngram_range=(1,3))
  pipe = Pipeline([('vectorizer', vectorizer ), ('tree', model)])
  pipe.fit(sentences_train_nfe, y_train_nfe)
  importance = pipe['tree'].feature_importances_
  features = pipe['vectorizer'].get_feature_names()
  features_importance_df = pd.DataFrame([features,importance])
  features_importance = features_importance_df.T .sort_values(1)
  features_importance.columns = ['feature', 'gini']
  features_importance.sort_values('gini', ascending=False)
  return features_importance



