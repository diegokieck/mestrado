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