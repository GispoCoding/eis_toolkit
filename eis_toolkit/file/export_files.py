
from typing import Optional, Any, Tuple
from pathlib import Path
import pandas as pd
import os
import joblib          # from joblib import dump, load
from eis_toolkit.exceptions import InvalidParameterValueException, FileReadWriteError, MissingFileOrPath

# *******************************
def _export_files(
    name: Optional[str] = 'ML',            # Name of Subject
    path: Optional[str] = None,
    validations: Optional[pd.DataFrame] = None, 
    crossvalidation: Optional[dict] = None,
    confusion_matrix: Optional[pd.DataFrame] = None,
    comparison: Optional[pd.DataFrame] = None,
    importance: Optional[pd.DataFrame] = None,
    kerasHistory: Optional[Any] = None,
    sklearnMl: Optional[Any] = None, 
    sklearnOhe: Optional[Any] = None,
    myFields: Optional[dict] = None,
    #myMetadata: Optional[dict] = None,
    kerasMl: Optional[Any] = None,
    kerasOhe: Optional[Any] = None,
    #nanmask: Optional[pd.DataFrame] = None, 
    decimalpoint_german: Optional[bool] = False,
    new_version: Optional[bool] = False,
) -> Tuple[dict]:

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
                try:
                    os.remove(os.path.abspath(filename+extension))
                except:
                    raise FileReadWriteError('Problem deleting file '+str(filename+extension))
        return filename

    # Main
    dt = {}             # dictionary of output files to use in input_files function
    if path is None:
        parent_dir = Path(__file__).parent
        path = parent_dir.joinpath(r'data')
    if not os.path.exists(path):
        raise MissingFileOrPath('path does not exists:' + str(path))
    dt['path'] = path

    if decimalpoint_german:
        decimal = ','
        separator = ';'
    else:
        decimal = '.'
        separator = ','

    if validations is not None:    # Validation
        # to csv
        file = create_file(path, name+'_validation', 'csv', new_version=new_version)
        #tmp =  pandas.DataFrame.from_dict(validations,orient='index')   # with dataframe
        # if decimalpoint_german:
        #     decimal = ','
        # else:
        #     decimal = '.'
        #tmp.to_csv(file+'.csv', sep=';', index = True, header= True, float_format='%00.4f', decimal = decimal)   # formatstring
        # tmp = tmp.astype('float',errors= 'ignore')     # str)
        try:
            validations.to_csv(file+'.csv', sep=separator, header=True, decimal=decimal, float_format='%00.5f')
        except:
           raise FileReadWriteError('Problem saving file ' +file+ '.csv')
        # with open(file+'.csv','w') as f:
        #     w = csv.writer(f)
        #     w.writerows(validations.items())

        # to json
        #import json
        file = create_file(path, name+'_validation','json', new_version=new_version)
        try:
            validations.to_json(file+'.json', orient='split', indent = 3)
        except:
           raise  FileReadWriteError('Problem saving file ' +file+'.json')
        # with open(file+'.json','w') as f:
        #     jsdata = json.dumps(validations, indent = 2)
        #     f.write(jsdata)
        #     f.close()

    if validations is not None and confusion_matrix is not None:
    # to csv
        file = create_file(path,name+'_confusion_matrix','csv',new_version=new_version)
        #tmp.to_csv(file, sep =';')
        try:
            confusion_matrix.to_csv(file+'.csv', sep=separator, index=True, header=True, decimal=decimal)   # float_format='%00.5f',
        except:
           raise  FileReadWriteError('Problem saving file ' +file+'.csv')
        # to json
        file = create_file(path, name+'_confusion_matrix', 'json', new_version=new_version)
        try:
            confusion_matrix.to_json(file+'.json', double_precision=5, orient='table', indent = 3)  # to string
        except:
            raise  FileReadWriteError('Problem saving file ' +file+'.json')   
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
        file = create_file(path, name+'_cross_validation', 'csv', new_version=new_version)
        tmp = pd.DataFrame.from_dict(ndict, orient='index')   # über dataframe

        #tmp.to_csv(file, sep =';')
        try:
            tmp.to_csv(file+'.csv', sep=';', index=True, header=True, float_format='%00.5f', decimal=decimal)   # formatstring
        except:
            raise  FileReadWriteError('Problem saving file ' +file+'.csv')
        # with open(file+'.csv','w') as f:
        #     for r in ndict:
        #                            # rows
        #     w = csv.writer(f)
        #     w.writerows(ndict.items())

        # to json
        #import json
        file = create_file(path, name+'_cross_validation', 'json', new_version=new_version)
        df = pd.DataFrame.from_dict(ndict)
        try:
            df.to_json(file+'.json', double_precision=5, orient='table', indent=3)  # to string
        except:
            raise  FileReadWriteError('Problem saving file ' +file+'.json')
        # with open(file+'.json','w') as f:               # von Dictionary: geht auch mit (zeilenumbruch)
        #     jsdata = json.dumps(ndict, indent = 2)
        #     f.write(jsdata)
        #     f.close()

    if comparison is not None:
    # to csv
        file = create_file(path, name+'_comparison', 'csv', new_version=new_version)
        #tmp.to_csv(file, sep =';')
        try:
            comparison.to_csv(file+'.csv', sep=separator, index=True, header=True, decimal=decimal)   # float_format='%00.5f',
        except:
            raise  FileReadWriteError('Problem saving file ' +file+'.csv')
        # to json
        file = create_file(path, name+'_comparison', 'json', new_version=new_version)
        try:
            comparison.to_json(file+'.json', double_precision=5, orient='table', indent = 3)  # to string
        except:
            raise  FileReadWriteError('Problem saving file ' +file+'.json')
    if importance is not None:
    # to csv
        file = create_file(path, name+'_importance', 'csv', new_version=new_version)
        #tmp.to_csv(file, sep =';')
        try:
            importance.to_csv(file+'.csv', sep=separator, index=True, header=True, decimal=decimal)   # float_format='%00.5f',
        except:
            raise  FileReadWriteError('Problem saving file ' +file+'.csv')
        # to json
        file = create_file(path, name+'_importance', 'json', new_version=new_version)
        try:
            importance.to_json(file+'.json', double_precision=5, orient='table', indent = 3)  # to string
        except:
            raise  FileReadWriteError('Problem saving file ' +file+'.json')
    if kerasHistory is not None:
        hist_df = pd.DataFrame(kerasHistory)
        # to csv
        file = create_file(path, name+'_kerashistory', 'csv', new_version=new_version)
        #tmp.to_csv(file, sep =';')
        try:
            hist_df.to_csv(file+'.csv', sep=separator, index=True, header=True, decimal=decimal)   # float_format='%00.5f',
        except:
            raise  FileReadWriteError('Problem saving file ' +file+'.csv')
        # to json
        file = create_file(path, name+'_kerashistory', 'json', new_version=new_version)
        try:
            hist_df.to_json(file+'.json', double_precision=5, orient='table', indent = 3)  # to string
        except:
            raise  FileReadWriteError('Problem saving file ' +file+'.json')
    if sklearnMl is not None:   # Model
        file = create_file(path,name+'_sklearnMl', 'mdl', new_version=new_version)
        dt['sklearnMl'] = file + '.mdl'
        try:
            joblib.dump(sklearnMl, file+'.mdl') # save the model
        except:
            raise  FileReadWriteError('Problem saving file ' +file+'.mdl')
        # clf = load('filename.joblib') # load and reuse the model

        # file = create_file(path, name+'sklearnMl','json')
        # with open("encoder", "wb") as f: 
        # pickle.dump(one_hot, f)

    if sklearnOhe is not None:   # OneHotEncoder
        file = create_file(path, name+'_sklearnOhe', 'ohe', new_version=new_version)
        dt['sklearnOhe'] = file + '.ohe'
        try:
            joblib.dump(sklearnOhe, file+'.ohe')
        except:
            raise  FileReadWriteError('Problem saving file ' +file+'.ohe')
    # fields
    if myFields is not None:    # Validation
        file = create_file(path,name+'_myFields', 'fld', new_version=new_version)
        dt['myFields'] = file + '.fld'
        import json
        try:
            with open(file+'.fld', 'w') as f:
                jsdata = json.dumps(myFields, indent=2)
                f.write(jsdata)
                f.close()
        except:
            raise  FileReadWriteError('Problem saving file ' +file+'.fld')
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
        file = create_file(path, name+'_kerasMl', 'h5', new_version=new_version)
        dt['kerasMl'] = file + '.h5'
        try:
            kerasMl.save(file+'.h5') # save the model
        except:
            raise  FileReadWriteError('Problem saving file ' +file+'.h5')
        # clf = load('filename.joblib') # load and reuse the model

        # file = create_file(path, name+'sklearnMl','json')
        # with open("encoder", "wb") as f: 
        # pickle.dump(one_hot, f)

    if kerasOhe is not None:   # OneHotEncoder
        file = create_file(path,name+'_kerasOhe', 'ohe', new_version=new_version)
        dict['kerasOhe'] = file+'.ohe'
        try:
            joblib.dump(kerasOhe, file+'.ohe')
        except:
            raise  FileReadWriteError('Problem saving file ' +file+'.ohe')
   
    return dt

# *******************************
def export_files(
    name: Optional[str] = 'ML',            # Vorgangsname
    path: Optional[str] = None,
    validations: Optional[pd.DataFrame] = None, 
    crossvalidation: Optional[dict] = None,
    confusion_matrix: Optional[pd.DataFrame] = None,
    comparison: Optional[pd.DataFrame] = None,
    importance: Optional[pd.DataFrame] = None,
    kerasHistory: Optional[Any] = None,
    sklearnMl: Optional[Any] = None,
    sklearnOhe: Optional[Any] = None,
    myFields: Optional[dict] = None,
    #myMetadata: Optional[dict] = None,
    kerasMl: Optional[Any] = None,
    kerasOhe: Optional[Any] = None,
    #nanmask: Optional[pd.DataFrame] = None,
    decimalpoint_german: Optional[bool] = False,
    new_version: Optional[bool] = False
) -> Tuple[dict]:

    """ 
        Writes files on disc, like the results of validation, fitted model or OneHotEncoder object.
    Args:
        - name (string, default 'ML'): Name of the ML-process (subject) will be used to build the filenames.
        - path (string, defalut './data'): Name of the output path. If the path does not exists, the function will raise an exception.
        At least one of the following arguments should be not None:
        - validation (DataFrame, default = None): Values of metrics according to the type of the model (classification or regression)
        - crossvalidation (dictionary, default = None): Content of the output file are validation values of each fold.
        - confusion_matrix (DataFrame,  default = None): Content of the output file is a table .
            Exist only for classifier estimators,
            (will be writte in a file just if validation is used).
        - importance (DataFrame,  default = None): Content of the output file is a table with values of importance for each feature.
        - kerasHistory (Dictionary, default = None): The Dictionary of history.history from keras fit
        - comparison (DataFrame,  default = None): List of all compared value pares: given, predicted 
        - sklearnMl (Model Object, default None): Content of the output file, type of the file: SKLAERN Model
        - sklearnOhe (OneHotEncoder Object, default None): Content of the output file, type of the file: SKLEARN OneHotEncoder
        - myFields (dictionary, default = None):  Content of the output file: columnslist with name and type of columns
        - kerasMl (Model Object, default None):  Tensorflor model to save as an file.
        - kerasOhe (OneHotEncoder Object, default None): content of the output file, type of the file: Ohe. In Case of Keras multiclass estimator 
        - decimalpoint_german (boolen, default False): True if the files above should get "," as decimal point and ";" in csv- files
        - new_version (boolen, default = False): If the file exists it schould be deletet (new_version = False) or a new version shold be created (new_version = True)
    Returns: 
        dictionary of the file names of sklearnMl, myOhe, myFields and myMetadata
    """
    # Argument evaluation
    fl = []
    if not ((isinstance(name, str)) or (name is None)):
        fl.append('Argument name is not a string and is not None')
    if not ((isinstance(path, str)) or (path is None)):
        fl.append('Argument path is not a string and is not None')
    if not ((isinstance(validations, pd.DataFrame)) or (validations is None)):
        fl.append('Argument validations is not a DataFrame and is not None')
    if not ((isinstance(comparison, pd.DataFrame)) or (comparison is None)):
        fl.append('Argument comparison is not a DataFrame and is not None')
    if not ((isinstance(confusion_matrix, pd.DataFrame)) or (confusion_matrix is None)):
        fl.append('Argument confusion_matrix is not a DataFrame and is not None')
    if not ((isinstance(importance, pd.DataFrame)) or (importance is  None)):
        fl.append('Argument importance is not a DataFrame and is not None')
    if not ((str(type(crossvalidation)) == "<class 'dict'>")  or (crossvalidation is None)):
    #if not ((isinstance(crossvalidation,dict)) or (crossvalidation is None)):
        fl.append('Argument crossvalidation is not a dictionary and is not None')
    if not ((str(type(myFields)) == "<class 'dict'>") or (myFields is None)):
    #if not ((isinstance(myFields,dict)) or (myFields is None)):
        fl.append('Argument myFields is not a dictionary and is not None')
    # if not ((isinstance(myMetadata,dict)) or (myMetadata is None)):
    #     fl.append('argument myMetadata is not a dictionary and is not None')
    t = sklearnMl.__class__.__name__           #t = isinstance(sklearnMl,(RandomForestClassifier,RandomForestRegressor,LogisticRegression))
    if not (t in ('RandomForestClassifier','RandomForestRegressor','LogisticRegression') or sklearnMl is None):
        fl.append('Argument sklearnMl ist not in instance of one of (RandomForestClassifier,RandomForestRegressor,LogisticRegression)')
        #raise InvalidParameterValueException ('***  sklearnMl ist not in instance of one of (RandomForestClassifier,RandomForestRegressor,LogisticRegression)')
    t = sklearnOhe.__class__.__name__ 
    if not (t in ('OneHotEncoder') or sklearnOhe is None):
        fl.append('Argument myOhe ist not in instance of one of OneHotEncoder')
    if not (kerasHistory.__class__.__name__ in ('History') or kerasHistory is None):
        fl.append('argument kerasHistory ist not in instance of one of History')
    t = kerasOhe.__class__.__name__ 
    if not (t in ('OneHotEncoder') or kerasOhe is None):
        fl.append('Argument kerasOhe ist not in instance of one of OneHotEncoder')
    if not (kerasMl.__class__.__name__ in ('Model') or kerasMl is None):
        fl.append('Argument kerasMl ist not an instance of one of Model')
    if not ((isinstance(decimalpoint_german, bool) or (decimalpoint_german is None)) and (isinstance(new_version,bool) or (decimalpoint_german is None))):
        fl.append('Argument decimalpoint_german or new_version are not boolean or are not None')
    if len(fl) > 0:
        raise InvalidParameterValueException (fl[0])

    return _export_files(
        name = name,
        path = path,
        validations  = validations, 
        crossvalidation = crossvalidation,
        confusion_matrix = confusion_matrix,
        comparison = comparison,
        importance = importance,
        kerasHistory = kerasHistory,
        sklearnMl = sklearnMl, 
        sklearnOhe = sklearnOhe,
        myFields = myFields,
        kerasMl = kerasMl,
        kerasOhe = kerasOhe,
        decimalpoint_german = decimalpoint_german,
        new_version = new_version,
    )

