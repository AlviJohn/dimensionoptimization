import pandas as pd
#import pandas_profiling as pp
import numpy as np
#import sweetviz as sv
from pycaret.regression import *
import shap
import seaborn as sns

raw_data= pd.read_csv('raw_data.csv')

fpindex_df = raw_data[['FP_Index','Mould_SD','Target_OD']]
contactarea_df = raw_data[['Contact_Area','Mould_SD','Target_OD']]

####################FP Index Model


pycaret_setup = setup(data = fpindex_df, target = 'FP_Index', 
                 train_size = 0.90,fold_shuffle=True,
                  session_id=123,imputation_type='iterative',numeric_features=['Mould_SD','Target_OD'],
                  normalize = True, transformation = False, fold=3,  
                  combine_rare_levels = True,log_plots = True,
                  remove_multicollinearity = False, multicollinearity_threshold = 0.95, 
                  log_experiment = True, experiment_name = 'Testing',n_jobs=1)

top10 = compare_models(n_select = 10)
results = pull()
print(results)

tuned_dt = tune_model(top10[0])
results=pull()
print(results)
results.to_csv('FPmodel.csv')

final_model = finalize_model(tuned_dt)
save_model(model, 'FP_Index_Model')


####################Contact Area Model

pycaret_setup = setup(data = contactarea_df, target = 'Contact_Area', 
                 train_size = 0.90,fold_shuffle=True,
                  session_id=123,imputation_type='iterative',numeric_features=['Mould_SD','Target_OD'],
                  normalize = True, transformation = False, fold=3,  
                  combine_rare_levels = True,log_plots = True,
                  remove_multicollinearity = False, multicollinearity_threshold = 0.95, 
                  log_experiment = True, experiment_name = 'Testing',n_jobs=1)

top10 = compare_models(n_select = 10)
results = pull()
print(results)

tuned_dt = tune_model(top10[0])
results=pull()
print(results)
results.to_csv('contactareamodel.csv')

final_model = finalize_model(tuned_dt)
save_model(model, 'Contactarea_Model')


