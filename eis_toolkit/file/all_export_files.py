
from typing import Optional, Any
from pathlib import Path
#import csv
import pandas as pd
import os
#import json
#import locale
#import pickle
import joblib          # from joblib import dump, load
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _all_export_files(
    name: Optional[str] = 'ML',            # Vorgangsname
    path: Optional[str] = None,
    validations: Optional[pd.DataFrame] = None, 
    crossvalidation: Optional[dict] = None,
    confusion_matrix: Optional[pd.DataFrame] = None,
    comparison: Optional[pd.DataFrame] = None,
    importance: Optional[pd.DataFrame] = None,
    kerasHistory: Optional[Any] = None,
    sklearnMl: Optional[Any] = None, 
    myOhe: Optional[Any] = None,
    myFields: Optional[dict] = None,
    #myMetadata: Optional[dict] = None,
    kerasMl: Optional[Any] = None,
    kerasOhe: Optional[Any] = None,
    #nanmask: Optional[pd.DataFrame] = None, 
    decimalpoint_german: Optional[bool] = False,
    new_version: Optional[bool] = False,
) -> dict:

    def create_file(
        path: str, 
        name: str, 
        extension: str,
        new_version: bool
    ) -> str:
        if len(extension) >0:
            extension = '.'+extension
        filenum = 1
        filename = os.path.join(path,name)
        if (os.path.exists(os.path.abspath(filename+extension))):
            if new_version:     # next file number
                while (os.path.exists(os.path.abspath(filename+str(filenum)+extension))):
                    filenum+=1
                return filename+str(filenum)   #open(filename+str(filenum)+extension,'w')
            else:               # file will be deletet
                os.remove(os.path.abspath(filename+extension))
        return filename

    # Argument evaluation
    fl = []
    if not ((isinstance(name,str)) or (name is None)):
        fl.append('argument name is not a string and is not None')
    if not ((isinstance(path, str)) or (path is None)):
        fl.append('argument path is not a string and is not None')
    if not ((isinstance(validations,pd.DataFrame)) or (validations is None)):
        fl.append('argument validations is not a DataFrame and is not None')
    if not ((isinstance(comparison,pd.DataFrame)) or (comparison is None)):
        fl.append('argument comparison is not a DataFrame and is not None')
    if not ((isinstance(confusion_matrix,pd.DataFrame)) or (confusion_matrix is None)):
        fl.append('argument confusion_matrix is not a DataFrame and is not None')
    if not ((isinstance(importance,pd.DataFrame)) or (importance is  None)):
        fl.append('argument importance is not a DataFrame and is not None')
    #(isinstance(crossvalidation,dict)
    if not (isinstance(crossvalidation,dict) or (crossvalidation is None)):
        fl.append('argument crossvalidation is not a dictionary and is not None')
    if not ((isinstance(myFields,dict)) or (myFields is None)):
        fl.append('argument myFields is not a dictionary and is not None')
    # if not ((isinstance(myMetadata,dict)) or (myMetadata is None)):
    #     fl.append('argument myMetadata is not a dictionary and is not None')
    t = sklearnMl.__class__.__name__           #t = isinstance(sklearnMl,(RandomForestClassifier,RandomForestRegressor,LogisticRegression))
    if not (t in ('RandomForestClassifier','RandomForestRegressor','LogisticRegression') or sklearnMl is None):
        fl.append('argument sklearnMl ist not in instance of one of (RandomForestClassifier,RandomForestRegressor,LogisticRegression)')
        #raise InvalidParameterValueException ('***  sklearnMl ist not in instance of one of (RandomForestClassifier,RandomForestRegressor,LogisticRegression)')
    t = myOhe.__class__.__name__ 
    if not (t in ('OneHotEncoder') or myOhe is None):
        fl.append('argument myOhe ist not in instance of one of OneHotEncoder')
    if not (kerasHistory.__class__.__name__ in ('History') or kerasHistory is None):
        fl.append('argument kerasHistory ist not in instance of one of History')
    t = kerasOhe.__class__.__name__ 
    if not (t in ('OneHotEncoder') or kerasOhe is None):
        fl.append('argument kerasOhe ist not in instance of one of OneHotEncoder')
    if not (kerasMl.__class__.__name__ in ('Model') or kerasMl is None):
        fl.append('argument kerasMl ist not an instance of one of Model')
    if not ((isinstance(decimalpoint_german,bool) or (decimalpoint_german is None)) and (isinstance(new_version,bool) or (decimalpoint_german is None))):
        fl.append('argument decimalpoint_german or new_version are not boolean or are not None')
    if len(fl) > 0:
        raise InvalidParameterValueException ('***  function all_export_files: ' + fl[0])

    # Main
    dt = {}
    if path is None:
        parent_dir = Path(__file__).parent
        path = parent_dir.joinpath(r'data')
    dt['path'] = path

    if decimalpoint_german:
        decimal = ','
        separator = ';'
    else:
        decimal = '.'
        separator = ','

    if validations is not None:    # Validation
        # to csv
        file = create_file(path,name+'_validation','csv',new_version=new_version)
        #tmp =  pandas.DataFrame.from_dict(validations,orient='index')   # with dataframe
        # if decimalpoint_german:
        #     decimal = ','
        # else:
        #     decimal = '.'
        #tmp.to_csv(file+'.csv', sep=';', index = True, header= True, float_format='%00.4f', decimal = decimal)   # formatstring
        # tmp = tmp.astype('float',errors= 'ignore')     # str)
        validations.to_csv(file+'.csv',sep=separator,header=True,decimal=decimal,float_format='%00.5f')
        # with open(file+'.csv','w') as f:
        #     w = csv.writer(f)
        #     w.writerows(validations.items())

        # to json
        #import json
        file = create_file(path,name+'_validation','json',new_version=new_version)
        validations.to_json(file+'.json',orient='split',indent = 3) 
        # with open(file+'.json','w') as f:
        #     jsdata = json.dumps(validations, indent = 2)
        #     f.write(jsdata)
        #     f.close()

    if validations is not None and confusion_matrix is not None:
    # to csv
        file = create_file(path,name+'_confusion_matrix','csv',new_version=new_version)
        #tmp.to_csv(file, sep =';')
        confusion_matrix.to_csv(file+'.csv',sep=separator,index=True,header=True,decimal=decimal)   # float_format='%00.5f',

        # to json
        file = create_file(path,name+'_confusion_matrix','json',new_version=new_version)
        confusion_matrix.to_json(file+'.json',double_precision=5,orient='table',indent = 3)  # to string
                                # orient = 'split', 'records', 'index', 'columns', 'values'
        # with open(file+'.json','w') as f:
        #     jsdata = json.dumps(ndict, indent = 2)
        #     f.write(jsdata)
        #     f.close()

    if crossvalidation is not None:    # cross validation
        # inner arrays to dictionaries
        ndict = {}
        for i in crossvalidation:         # i: keys
            nfolds = {}
            zl = 0
            for j in crossvalidation[i]:                 # array of the folds
                nfolds['fold'+(zl+1).__str__()] = j
                zl += 1
            ndict[i] = nfolds

        # to csv
        file = create_file(path,name+'_cross_validation','csv',new_version=new_version)
        tmp = pd.DataFrame.from_dict(ndict,orient='index')   # über dataframe
        #tmp.to_csv(file, sep =';')
        tmp.to_csv(file+'.csv', sep=';',index=True,header=True,float_format='%00.5f',decimal=decimal)   # formatstring
        # with open(file+'.csv','w') as f:
        #     for r in ndict:
        #                            # rows
        #     w = csv.writer(f)
        #     w.writerows(ndict.items())

        # to json
        #import json
        file = create_file(path,name+'_cross_validation','json',new_version=new_version)
        df = pd.DataFrame.from_dict(ndict)
        df.to_json(file+'.json',double_precision=5,orient='table',indent=3)  # to string
        # with open(file+'.json','w') as f:               # von Dictionary: geht auch mit (zeilenumbruch)
        #     jsdata = json.dumps(ndict, indent = 2)
        #     f.write(jsdata)
        #     f.close()

    if comparison is not None:
    # to csv
        file = create_file(path,name+'_comparison','csv',new_version=new_version)
        #tmp.to_csv(file, sep =';')
        comparison.to_csv(file+'.csv',sep=separator,index=True,header=True,decimal=decimal)   # float_format='%00.5f',

        # to json
        file = create_file(path,name+'_comparison','json',new_version=new_version)
        comparison.to_json(file+'.json',double_precision=5,orient='table',indent = 3)  # to string
 
    if importance is not None:
    # to csv
        file = create_file(path,name+'_importance','csv',new_version=new_version)
        #tmp.to_csv(file, sep =';')
        importance.to_csv(file+'.csv',sep=separator,index=True,header=True,decimal=decimal)   # float_format='%00.5f',

        # to json
        file = create_file(path,name+'_importance','json',new_version=new_version)
        importance.to_json(file+'.json',double_precision=5,orient='table',indent = 3)  # to string

    if kerasHistory is not None:
        hist_df = pd.DataFrame(kerasHistory)
        # to csv
        file = create_file(path,name+'_kerashistory','csv',new_version=new_version)
        #tmp.to_csv(file, sep =';')
        hist_df.to_csv(file+'.csv',sep=separator,index=True,header=True,decimal=decimal)   # float_format='%00.5f',

        # to json
        file = create_file(path,name+'_kerashistory','json',new_version=new_version)
        hist_df.to_json(file+'.json',double_precision=5,orient='table',indent = 3)  # to string
 
    if sklearnMl is not None:   # Model
        file = create_file(path,name+'_sklearnMl','mdl',new_version=new_version)
        dt['sklearnMl'] = file+'.mdl'
        joblib.dump(sklearnMl,file+'.mdl') # save the model
        # clf = load('filename.joblib') # load and reuse the model

        # file = create_file(path, name+'sklearnMl','json')
        # with open("encoder", "wb") as f: 
        # pickle.dump(one_hot, f)

    if myOhe is not None:   # OneHotEncoder
        file = create_file(path,name+'_myOhe','ohe',new_version=new_version)
        dt['myOhe'] = file+'.ohe'
        joblib.dump(myOhe,file+'.ohe')
   
    # fields
    if myFields is not None:    # Validation
        file = create_file(path,name+'_myFields','fld',new_version=new_version)
        dt['myFields'] = file+'.fld'
        import json
        with open(file+'.fld','w') as f:
            jsdata = json.dumps(myFields,indent=2)
            f.write(jsdata)
            f.close()
        # Read data from file:
        # data = json.load( open( "file_name.json" ) )

    # # metadata
    # if myMetadata is not None:    # Validation
    #     file = create_file(path,name+'_myMetadata','mtd',new_version=new_version)
    #     dt['myMetadata'] = file+'.mtd'
    #     joblib.dump(myMetadata,file+'.mtd')
    #     # import json
    #     # with open(file+'.mtd','w') as f:
    #     #     jsdata = json.dumps(myMetadata, indent = 2)
    #     #     f.write(jsdata)
    #     #     f.close()

    #     # Read data from file:
    #     # data = json.load( open( "file_name.json" ) )

    if kerasMl is not None:   # Model
        file = create_file(path,name+'_kerasMl','h5',new_version=new_version)
        dt['kerasMl'] = file+'.h5'
        kerasMl.save(file+'.h5') # save the model
        # clf = load('filename.joblib') # load and reuse the model

        # file = create_file(path, name+'sklearnMl','json')
        # with open("encoder", "wb") as f: 
        # pickle.dump(one_hot, f)

    if kerasOhe is not None:   # OneHotEncoder
        file = create_file(path,name+'_kerasOhe','ohe',new_version=new_version)
        dict['kerasOhe'] = file+'.ohe'
        joblib.dump(kerasOhe,file+'.ohe')
   
    return dt

# *******************************
def all_export_files(
    name: Optional[str] = 'ML',            # Vorgangsname
    path: Optional[str] = None,
    validations: Optional[pd.DataFrame] = None, 
    crossvalidation: Optional[dict] = None,
    confusion_matrix: Optional[pd.DataFrame] = None,
    comparison: Optional[pd.DataFrame] = None,
    importance: Optional[pd.DataFrame] = None,
    kerasHistory: Optional[Any] = None,
    sklearnMl: Optional[Any] = None,
    myOhe: Optional[Any] = None,
    myFields: Optional[dict] = None,
    #myMetadata: Optional[dict] = None,
    kerasMl: Optional[Any] = None,
    kerasOhe: Optional[Any] = None,
    #nanmask: Optional[pd.DataFrame] = None,
    decimalpoint_german: Optional[bool] = False,
    new_version: Optional[bool] = False
) -> dict:

    """ 
        writes csv and json files on disc 
    Args:
        - name (string, default 'ML'): Name of the ML-process
        - path (string, defalut './data'): Name of the output path
        at least one of the following arguments should be not None
        - testsplit (DatasFrame, default None): content of the output file: validation values from test_split
        - crossvalidation (dictionary, default = None): content of the output file are validation values of each fold
        - confusion_matrix (DataFrame,  default = None): content of the output file is a table 
            exist only for classifier estimators,
            (will be writte in a file just isf testsplit is used)
        - importance (DataFrame,  default = None): content of the output file is a table with values of importance for each feature
        - kerasHistory (Dictionary, default = None): the Dictionary of history.history from keras fit
        - comparison (DataFrame,  default = None): list of all compared values: given, predicted
        - sklearnMl (Model Object, default None): content of the output file, type of the file: Model
        - myOhe (OneHotEncoder Object, default None): content of the output file, type of the file: Ohe
        - myFields (dictionary, default = None):  content of the output file: columnslist with name and type of columns
        - kerasMl (Model Object, default None):  Tensorflor model zu save as an file
        - kerasOhe (OneHotEncoder Object, default None): content of the output file, type of the file: Ohe. In Case of Keras multiclass estimator 
        - decimalpoint_german (boolen default False): True if the files above should get "," as decimal point and ";" in csv- files
        - new_version (boolen, default = False): if the file exists it schould be deletet (new_version = False) or a new version shold be created
    Returns: 
        dictionary of the files names für sklearnMl, myOhe, myFields and myMetadata
    """

    dict = _all_export_files(
        name = name,
        path = path,
        validations  = validations, 
        crossvalidation = crossvalidation,
        confusion_matrix = confusion_matrix,
        comparison = comparison,
        importance = importance,
        kerasHistory = kerasHistory,
        sklearnMl = sklearnMl, 
        myOhe = myOhe,
        myFields = myFields,
        #myMetadata = myMetadata,
        kerasMl = kerasMl,
        kerasOhe = kerasOhe,
        decimalpoint_german = decimalpoint_german,
        new_version = new_version,
    )

    return dict
