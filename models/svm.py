import pandas as pd
import numpy as np
import sklearn as sk
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, pyll
from google.colab import drive

import hyperopt

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

svm_space = {
    'kernel' : hp.choice('a', ['linear', 'poly', 'rbf', 'sigmoid']),
    'max_df' : hp.choice('b', [1.0, 0.5, 0.75]),
    'min_df' : hp.choice('c', [1,0.1, 0.01, 0.001,0.0001 ]),
    'ngram' : hp.choice('d', [(1,1), (1,2), (1,3)]),
    'max_features' : hp.choice('e',[None])
}

svm_space_list ={
    'kernel' : hp.choice('a', ['linear', 'poly', 'rbf', 'sigmoid']),
    'max_df' : hp.choice('b', [1.0, 0.5, 0.75]),
    'min_df' : hp.choice('c', [1,0.1, 0.01, 0.001,0.0001 ]),
    'ngram' : hp.choice('d', [(1,1), (1,2), (1,3)]),
    'max_features' : hp.choice('e',[None])

}

def get_model(args):
  model = SVC(kernel=args['kernel'])
  #define model   
  model = SVC(kernel=args['kernel'])
  #define vectorizer
  vectorizer = TfidfVectorizer(
      max_df= args['max_df'],
      min_df= args['min_df'],
      ngram_range = args['ngram'],
      max_features= args['max_features']
  )
  #define pipe
  pipe = Pipeline([('vectorizer', vectorizer ), ('svc', model)])
  return pipe