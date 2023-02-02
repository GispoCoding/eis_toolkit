
#### 
# Dictionaries werden als json und csv-file so wie modelle und onehot-Encoderexportiert
# abgucken in advangeo

from typing import Optional, Any
from pathlib import Path
#import csv
import pandas
import os
#import json
#import locale
#import pickle
import joblib          # from joblib import dump, load
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _export_files(
    name: Optional[str] = 'ML',            # Vorgangsname
    path: Optional[str] = None,
    validations: Optional[pandas.DataFrame] = None, 
    crossvalidation: Optional[dict] = None,
    confusion_matrix: Optional[pandas.DataFrame] = None,
    comparison: Optional[pandas.DataFrame] = None,
    myML: Optional[Any] = None, 
    myOhe: Optional[Any] = None,
    myFields: Optional[dict] = None,
    myMetadata: Optional[dict] = None,
    # importance,
    #nanmask: Optional[pd.DataFrame] = None 
    decimalpoint_german: Optional[bool] = False
) -> dict: 
    def create_file(
        path: str, 
        name: str, 
        extension: str,
    ) -> str:
    
        filenum = 1
        filename = os.path.join(path,name)
        if (os.path.exists(os.path.abspath(filename+'.'+extension))):
            while (os.path.exists(os.path.abspath(filename+str(filenum)+'.'+extension))):
                filenum+=1
            return filename+str(filenum)   #open(filename+str(filenum)+'.'+extension,'w')
        return filename

    # Main
    dict = {}
    if path is None:
        parent_dir = Path(__file__).parent
        path = parent_dir.joinpath(r'data')
    dict['path'] = path 

    if decimalpoint_german:
        decimal = ','
        separator = ';'
    else:
        decimal = '.'
        separator = ','

    if validations is not None:    # Validation
        # to csv
        file = create_file(path,name+'_validation','csv')
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
        import json
        file = create_file(path,name+'_validation','json')
        validations.to_json(file+'.json',orient='split',indent = 3) 
        # with open(file+'.json','w') as f:
        #     jsdata = json.dumps(validations, indent = 2)
        #     f.write(jsdata)
        #     f.close()

    if validations is not None and confusion_matrix is not None:
    # to csv
        file = create_file(path,name+'_confusion_matrix','csv')
        #tmp.to_csv(file, sep =';')
        confusion_matrix.to_csv(file+'.csv',sep=separator,index=True,header=True,decimal=decimal)   # float_format='%00.5f',

        # to json
        file = create_file(path,name+'_confusion_matrix','json')
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
        file = create_file(path,name+'_cross_validation','csv')
        tmp = pandas.DataFrame.from_dict(ndict,orient='index')   # über dataframe
        #tmp.to_csv(file, sep =';')
        tmp.to_csv(file+'.csv', sep=';',index=True,header=True,float_format='%00.5f',decimal=decimal)   # formatstring
        # with open(file+'.csv','w') as f:
        #     for r in ndict:
        #                            # rows
        #     w = csv.writer(f)
        #     w.writerows(ndict.items())

        # to json
        import json
        file = create_file(path,name+'_cross_validation','json')
        df = pandas.DataFrame.from_dict(ndict)
        df.to_json(file+'.json',double_precision=5,orient='table',indent=3)  # to string
        # with open(file+'.json','w') as f:               # von Dictionary: geht auch mit (zeilenumbruch)
        #     jsdata = json.dumps(ndict, indent = 2)
        #     f.write(jsdata)
        #     f.close()

    if comparison is not None:
    # to csv
        file = create_file(path,name+'_comparison','csv')
        #tmp.to_csv(file, sep =';')
        comparison.to_csv(file+'.csv',sep=separator,index=True,header=True,decimal=decimal)   # float_format='%00.5f',

        # to json
        file = create_file(path,name+'_comparison','json')
        comparison.to_json(file+'.json',double_precision=5,orient='table',indent = 3)  # to string
 
    if myML is not None:   # Model
        file = create_file(path,name+'_myML','mdl')
        dict['myML'] = file+'.mdl'
        joblib.dump(myML,file+'.mdl') # save the model
        # clf = load('filename.joblib') # load and reuse the model

        # file = create_file(path, name+'myML','json')
        # with open("encoder", "wb") as f: 
        # pickle.dump(one_hot, f)

    if myOhe is not None:   # OneHotEncoder
        file = create_file(path,name+'_myOhe','ohe')
        dict['myOhe'] = file+'.ohe'
        joblib.dump(myOhe,file+'.ohe')
   
    # fields
    if myFields is not None:    # Validation
        file = create_file(path,name+'_myFields','fld')
        dict['myFields'] = file+'.fld'
        import json
        with open(file+'.fld','w') as f:
            jsdata = json.dumps(myFields,indent=2)
            f.write(jsdata)
            f.close()
        # Read data from file:
        # data = json.load( open( "file_name.json" ) )

    # metadata
    if myMetadata is not None:    # Validation
        file = create_file(path,name+'_myMetadata','mtd')
        dict['myMetadata'] = file+'.mtd'
        joblib.dump(myMetadata,file+'.mtd')
        # import json
        # with open(file+'.mtd','w') as f:
        #     jsdata = json.dumps(myMetadata, indent = 2)
        #     f.write(jsdata)
        #     f.close()

        # Read data from file:
        # data = json.load( open( "file_name.json" ) )

    return dict

# *******************************
def export_files(
    name: Optional[str] = 'ML',            # Vorgangsname
    path: Optional[str] = None,
    validations: Optional[pandas.DataFrame] = None, 
    crossvalidation: Optional[dict] = None,
    confusion_matrix: Optional[pandas.DataFrame] = None,
    # importance,
    comparison: Optional[pandas.DataFrame] = None,
    myML: Optional[Any] = None, 
    myOhe: Optional[Any] = None,
    myFields: Optional[dict] = None,
    myMetadata: Optional[dict] = None,
    #nanmask: Optional[pd.DataFrame] = None 
    decimalpoint_german: Optional[bool] = False
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
        - comparison (DataFrame,  default = None): list of all compared values: given, predicted
        - myML (Model Object, default None): content of the output file, type of the file: Model
        - myOhe (OneHotEncoder Object, default None): content of the output file, type of the file: Ohe
        - myFields (dictionary, default = None):  content of the output file: columnslist with name and type of columns
        - myMetadata (default = None): content of the output file, Grid-Metadata  
    Returns: 
        dictionary of the files names für myML, myOhe, myFields and myMetadata
    """

    dict = _export_files(
        name = name,
        path = path,
        validations  = validations, 
        crossvalidation = crossvalidation,
        confusion_matrix = confusion_matrix,
        comparison = comparison,
        myML = myML, 
        myOhe = myOhe,
        myFields = myFields,
        myMetadata = myMetadata,
        decimalpoint_german = decimalpoint_german
    )

    return dict

