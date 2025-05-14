import numpy as np
import pandas as pd
import matplotlib.patheffects as path_effects
import matplotlib
import matplotlib.pyplot as plt

def get_right_units_vil(vil):
    """they scaled VIL weird, so this unscales it"""
    tmp = np.zeros(vil.shape)
    idx = np.where(vil <=5)
    tmp[idx] = 0
    idx = np.where((vil>5)*(vil <= 18))
    tmp[idx] = (vil[idx] -2)/90.66
    idx = np.where(vil>18)
    tmp[idx] = np.exp((vil[idx] - 83.9)/38.9)
    return tmp

def adjust_keys(df,keyadd,dask=False,dropevent=False):
    if dask:
        keys = df.columns
        newkeys = []
        newkeys.append('dtime')
        newkeys = newkeys + list(keys[1:-1]+keyadd)
        newkeys.append(keys[-1])
    else:
        keys = df.keys()
        newkeys = list(keys[:-1]+keyadd)
        newkeys.append(keys[-1])
        
    df.columns = newkeys
    if dropevent:
        df = df.drop(columns='event')
        
    if dask:
        df['dtime'] = df['dtime'].astype(np.datetime64)
    return df 

def clear_nan(X,y):
    tmp = np.hstack([X,y.reshape([y.shape[0],1])])
    df_tmp = pd.DataFrame(tmp)
    df_tmp = df_tmp.dropna(how='any')
    tmp = df_tmp.to_numpy()
    X = tmp[:,:-1]
    y = tmp[:,-1:]
    y = np.asarray(y.squeeze(),dtype=int)
    return X,y

def load_n_combine_df(date = '2019-03-01',features_to_keep=np.arange(0,36,1),class_labels=True,dropzeros=False):
    df_ir = pd.read_csv('IR_stats_master.csv',index_col=0,low_memory=False,parse_dates=True)
    df_wv = pd.read_csv('WV_stats_master.csv',index_col=0,low_memory=False,parse_dates=True)
    df_vis = pd.read_csv('VIS_stats_master.csv',index_col=0,low_memory=False,parse_dates=True)
    df_vil = pd.read_csv('VIL_stats_master.csv',index_col=0,low_memory=False,parse_dates=True)
    df_li = pd.read_csv('LI_stats_master.csv',index_col=0,low_memory=False,parse_dates=True)

    #Nos deshacemos del outlier
    df_wv = df_wv.where(df_wv.q000 > -10000)
    
    if dropzeros:
        df_li = df_li.where(df_li.c >= 1)
    
    #Quitar NaNs:
    idx_keep = np.where(~df_vis.isna().all(axis=1).values)[0]
    df_ir = df_ir.iloc[idx_keep]
    df_wv = df_wv.iloc[idx_keep]
    df_vis = df_vis.iloc[idx_keep]
    df_vil = df_vil.iloc[idx_keep]
    df_li = df_li.iloc[idx_keep]

    #Hacer que los indices esten en orden:
    df_ir = df_ir.sort_index()
    df_wv = df_wv.sort_index()
    df_vis = df_vis.sort_index()
    df_vil = df_vil.sort_index()
    df_li = df_li.sort_index()

 #Adjustar las claves:
    df_ir = adjust_keys(df_ir,'_ir')
    df_wv = adjust_keys(df_wv,'_wv')
    df_vis = adjust_keys(df_vis,'_vi')
    df_vil = adjust_keys(df_vil,'_vl')
    df_li = adjust_keys(df_li,'_li')

    #Quitar la columna de eventos:
    df_ir= df_ir.drop(columns='event')
    df_wv= df_wv.drop(columns='event')
    df_vis= df_vis.drop(columns='event')
    df_vil= df_vil.drop(columns='event')
    df_li = df_li.drop(columns='event')
    
    #Hacer slices en el tiempo para el subconjunto de entrenamiento y test: 
    train_slice = slice('2017-01-01',date) #'2017-01-01','2019-06-01'
    other_slice = slice(date,'2019-12-31') #'2019-06-01','2019-12-31'

    df_ir_tr = df_ir[train_slice]
    df_ir_ot = df_ir[other_slice]
    df_wv_tr = df_wv[train_slice]
    df_wv_ot = df_wv[other_slice]
    df_vis_tr = df_vis[train_slice]
    df_vis_ot = df_vis[other_slice]
    df_vil_tr = df_vil[train_slice]
    df_vil_ot = df_vil[other_slice]
    df_li_tr = df_li[train_slice]
    df_li_ot = df_li[other_slice]

# Enviar el resto de semanas a validacion o test:
    va = np.arange(22,52,2)
    te = np.arange(23,53,2)
    dtime = pd.to_datetime(df_ir_ot.index)

    idx_v = np.array([],dtype=int)
    for v in va:
        tmp = np.where(dtime.isocalendar().week == v)[0]
        if len(tmp) == 0:
            continue
        else:
            idx_v = np.append(idx_v,tmp)

    idx_t = np.array([],dtype=int)      
    for t in te:
        tmp = np.where(dtime.isocalendar().week == t)[0]
        if len(tmp) == 0:
            continue
        else:
            idx_t = np.append(idx_t,tmp)


    df_ir_va = df_ir_ot.iloc[idx_v]
    df_ir_te = df_ir_ot.iloc[idx_t]
    df_wv_va = df_wv_ot.iloc[idx_v]
    df_wv_te = df_wv_ot.iloc[idx_t]
    df_vis_va = df_vis_ot.iloc[idx_v]
    df_vis_te = df_vis_ot.iloc[idx_t]
    df_vil_va = df_vil_ot.iloc[idx_v]
    df_vil_te = df_vil_ot.iloc[idx_t]
    df_li_va = df_li_ot.iloc[idx_v]
    df_li_te = df_li_ot.iloc[idx_t]

    X_train = np.hstack([df_ir_tr.to_numpy()*1e-2,df_wv_tr.to_numpy()*1e-2,df_vis_tr.to_numpy()*1e-4,get_right_units_vil(df_vil_tr.to_numpy())])
    X_validate = np.hstack([df_ir_va.to_numpy()*1e-2,df_wv_va.to_numpy()*1e-2,df_vis_va.to_numpy()*1e-4,get_right_units_vil(df_vil_va.to_numpy())])
    X_test= np.hstack([df_ir_te.to_numpy()*1e-2,df_wv_te.to_numpy()*1e-2,df_vis_te.to_numpy()*1e-4,get_right_units_vil(df_vil_te.to_numpy())])

    #Escoger:
    X_train = X_train[:,features_to_keep]
    X_validate = X_validate[:,features_to_keep]
    X_test = X_test[:,features_to_keep]

    #Hacer etiquetas de clase:
    if class_labels:
        y_train = np.zeros(X_train.shape[0],dtype=int)
        y_train[np.where(df_li_tr.c_li.values >= 1)] = 1

        y_validate = np.zeros(X_validate.shape[0],dtype=int)
        y_validate[np.where(df_li_va.c_li.values >= 1)] = 1

        y_test = np.zeros(X_test.shape[0],dtype=int)
        y_test[np.where(df_li_te.c_li.values >= 1)] = 1
    else:
        y_train = df_li_tr.c_li.values
        y_validate = df_li_va.c_li.values
        y_test = df_li_te.c_li.values

 #Limpiar NaNs: 
    X_train,y_train = clear_nan(X_train,y_train)
    X_validate,y_validate = clear_nan(X_validate,y_validate)
    X_test,y_test = clear_nan(X_test,y_test)
    
    return (X_train,y_train),(X_validate,y_validate),(X_test,y_test)


#########################################################
#######MODIFICACION CON DIFERENTES SPLITS TEMPORALES#####
#########################################################

def load_n_combine_mod_df(features_to_keep=np.arange(0,36,1),class_labels=True,dropzeros=False):
    df_ir = pd.read_csv('IR_stats_master.csv',index_col=0,low_memory=False,parse_dates=True)
    df_wv = pd.read_csv('WV_stats_master.csv',index_col=0,low_memory=False,parse_dates=True)
    df_vis = pd.read_csv('VIS_stats_master.csv',index_col=0,low_memory=False,parse_dates=True)
    df_vil = pd.read_csv('VIL_stats_master.csv',index_col=0,low_memory=False,parse_dates=True)
    df_li = pd.read_csv('LI_stats_master.csv',index_col=0,low_memory=False,parse_dates=True)

    #Creamos las listas de datasets X_train...:
    list_X_train=[]
    list_X_validate=[]
    list_X_test=[]
    list_y_train=[]
    list_y_validate=[]
    list_y_test=[]

    #Nos deshacemos del outlier
    df_wv = df_wv.where(df_wv.q000 > -10000)
    
    if dropzeros:
        df_li = df_li.where(df_li.c >= 1)
    
    #Quitar NaNs:
    idx_keep = np.where(~df_vis.isna().all(axis=1).values)[0]
    df_ir = df_ir.iloc[idx_keep]
    df_wv = df_wv.iloc[idx_keep]
    df_vis = df_vis.iloc[idx_keep]
    df_vil = df_vil.iloc[idx_keep]
    df_li = df_li.iloc[idx_keep]

    #Hacer que los indices esten en orden:
    df_ir = df_ir.sort_index()
    df_wv = df_wv.sort_index()
    df_vis = df_vis.sort_index()
    df_vil = df_vil.sort_index()
    df_li = df_li.sort_index()

 #Adjustar las claves:
    df_ir = adjust_keys(df_ir,'_ir')
    df_wv = adjust_keys(df_wv,'_wv')
    df_vis = adjust_keys(df_vis,'_vi')
    df_vil = adjust_keys(df_vil,'_vl')
    df_li = adjust_keys(df_li,'_li')

    #Quitar la columna de eventos:
    df_ir= df_ir.drop(columns='event')
    df_wv= df_wv.drop(columns='event')
    df_vis= df_vis.drop(columns='event')
    df_vil= df_vil.drop(columns='event')
    df_li = df_li.drop(columns='event')

    #Coger cada fecha de la lista:
    list_dates = ['2017-06-01','2017-12-01','2018-04-01','2018-05-01','2018-06-01','2018-12-01','2019-02-01','2019-04-01','2019-06-01']

    for date in list_dates:

        #Hacemos slices en el tiempo para el subconjunto de entrenamiento y test:
        train_slice = slice('2017-01-01',date) #'2017-01-01','2019-06-01'
        other_slice = slice(date,'2019-12-31') #'2019-06-01','2019-12-31'

        df_ir_tr = df_ir[train_slice]
        df_ir_ot = df_ir[other_slice]
        df_wv_tr = df_wv[train_slice]
        df_wv_ot = df_wv[other_slice]
        df_vis_tr = df_vis[train_slice]
        df_vis_ot = df_vis[other_slice]
        df_vil_tr = df_vil[train_slice]
        df_vil_ot = df_vil[other_slice]
        df_li_tr = df_li[train_slice]
        df_li_ot = df_li[other_slice]

    # Enviar el resto de semanas a validacion o test:
        va = np.arange(22,52,2)
        te = np.arange(23,53,2)
        dtime = pd.to_datetime(df_ir_ot.index)
    
        idx_v = np.array([],dtype=int)
        for v in va:
            tmp = np.where(dtime.isocalendar().week == v)[0]
            if len(tmp) == 0:
                continue
            else:
                idx_v = np.append(idx_v,tmp)
    
        idx_t = np.array([],dtype=int)      
        for t in te:
            tmp = np.where(dtime.isocalendar().week == t)[0]
            if len(tmp) == 0:
                continue
            else:
                idx_t = np.append(idx_t,tmp)


        df_ir_va = df_ir_ot.iloc[idx_v]
        df_ir_te = df_ir_ot.iloc[idx_t]
        df_wv_va = df_wv_ot.iloc[idx_v]
        df_wv_te = df_wv_ot.iloc[idx_t]
        df_vis_va = df_vis_ot.iloc[idx_v]
        df_vis_te = df_vis_ot.iloc[idx_t]
        df_vil_va = df_vil_ot.iloc[idx_v]
        df_vil_te = df_vil_ot.iloc[idx_t]
        df_li_va = df_li_ot.iloc[idx_v]
        df_li_te = df_li_ot.iloc[idx_t]
    
        X_train = np.hstack([df_ir_tr.to_numpy()*1e-2,df_wv_tr.to_numpy()*1e-2,df_vis_tr.to_numpy()*1e-4,get_right_units_vil(df_vil_tr.to_numpy())])
        X_validate = np.hstack([df_ir_va.to_numpy()*1e-2,df_wv_va.to_numpy()*1e-2,df_vis_va.to_numpy()*1e-4,get_right_units_vil(df_vil_va.to_numpy())])
        X_test= np.hstack([df_ir_te.to_numpy()*1e-2,df_wv_te.to_numpy()*1e-2,df_vis_te.to_numpy()*1e-4,get_right_units_vil(df_vil_te.to_numpy())])

        #Escoger:
        X_train = X_train[:,features_to_keep]
        X_validate = X_validate[:,features_to_keep]
        X_test = X_test[:,features_to_keep]

        #Hacer etiquetas de clase:
        if class_labels:
            y_train = np.zeros(X_train.shape[0],dtype=int)
            y_train[np.where(df_li_tr.c_li.values >= 1)] = 1
    
            y_validate = np.zeros(X_validate.shape[0],dtype=int)
            y_validate[np.where(df_li_va.c_li.values >= 1)] = 1
    
            y_test = np.zeros(X_test.shape[0],dtype=int)
            y_test[np.where(df_li_te.c_li.values >= 1)] = 1
        else:
            y_train = df_li_tr.c_li.values
            y_validate = df_li_va.c_li.values
            y_test = df_li_te.c_li.values

     #Limpiar NaNs: 
        X_train,y_train = clear_nan(X_train,y_train)
        X_validate,y_validate = clear_nan(X_validate,y_validate)
        X_test,y_test = clear_nan(X_test,y_test)

        #Hacemos un append a las listas:
        list_X_train.append(X_train)
        list_X_validate.append(X_validate)
        list_X_test.append(X_test)
        list_y_train.append(y_train)
        list_y_validate.append(y_validate)
        list_y_test.append(y_test)
        
    
    return (list_X_train,list_y_train),(list_X_validate,list_y_validate),(list_X_test,list_y_test)
