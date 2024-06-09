import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error


import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.feature_selection import  SelectKBest, RFE, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import  DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from collections import Counter



def fit_model(df, vars=None, clf="rf"):
    """
    Funcion para el entrenamiento de los modelos
    
    Parametros:
    --------------
        vars: Variables a utilizar para el modelo
        clf: Tipo de algoritmo a utilizar. ("rf", "gb", "svr")
        
    """   

    # df = df.sample(10000)  
    df = df.copy()  
    
    # Definir la variable objetivo
    target = "Costo_por_evento"
    
    to_drop = ["Asegurado_Id", "Poliza_Asegurado_Id", "Valor_Pagado", target]
    
    
    if vars:
        X = df[vars]
    else:
        X = df.drop(to_drop, axis=1)
    
    # Convertir la variable objetivo a horas
    y = df[target]
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configurar el preprocesamiento de variables categóricas y numéricas
    if clf == "svr":
        cat_processor = OneHotEncoder()
    else:
        cat_processor = OrdinalEncoder()
    
    
    # Preprocesador de variables  numericas
    num_processor = StandardScaler()
    
    # DataFrame de variables numericas  
    num_vals = X._get_numeric_data().columns.tolist()
    # DataFrame de variables categoricas
    cat_vals = X.select_dtypes("object").columns.tolist()
      
    
    # Combinar preprocesadores de variables
    processor = ColumnTransformer(transformers=[("cat", cat_processor, cat_vals), ("num", num_processor, num_vals)])
    
    # Seleccionar el modelo según el clasificador especificado
    if clf == "svr":
        model = make_pipeline(processor, SVR())
    elif clf == "rf":
        model = make_pipeline(processor, RandomForestRegressor(random_state=42))
    elif clf == "gb":
        model = make_pipeline(processor, GradientBoostingRegressor(random_state=42))
    
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    # Verificar si existe el directorio 'models', si no, crearlo
    # if os.path.exists('models'):
    #     pass
    # else:
    #     os.makedirs("models")
    
    # Retornar el modelo entrenado
    return model, X_test, y_test

def feature_sel(df, num_feat_kbest=7, num_rfe=10, sample=None, plot_metric="mutual_info", plot=False):
    """
    Funcion para la seleccion de las variables a utilizar

    Parametros:
    --------------
        num_feat_kbest: Variables a utilizar para de la seleccion por Kbest
        num_rfe: Variables a tener en cuenta con el metodo RFE
        plot_metric: Metrica por la cual se ordernaran los graficos
        plot: True si se quiere visualizar los resultados obtenidos

    """
    
    # df = df.sample(10000)  
    df_vars = df.copy()  
    
    # Definir la variable objetivo
    target = "Costo_por_evento"
    
    to_drop = ["Asegurado_Id", "Poliza_Asegurado_Id", target]
    
    # X = df.drop(to_drop, axis=1)
    

    X = df_vars.drop(to_drop, axis=1)
    
    # Convertir la variable objetivo a horas
    y = df_vars[target]

    # Division de datos de evaluacion y entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Preprocesador de variables categoricas
    cat_processor = OrdinalEncoder()
    # Preprocesador de variables numericas
    num_processor = MinMaxScaler()

    # DataFrame de variables numericas
    num_vals = X._get_numeric_data().columns.tolist()
    # DataFrame de variables categoricas
    cat_vals = X.select_dtypes("object").columns.tolist()

    # Union de procesadores
    processor = ColumnTransformer(transformers=[("cat", cat_processor, cat_vals), ("num", num_processor, num_vals)])

    # Dicionarion para guardar resultados del algoritmo Kbest (una columna por metrica)
    vars = {"f_regression":[], "mutual_info":[]}

    # Evaluacion de algoritmo Kbest con diferentes metricas y diferentes k
    for m in (f_regression, mutual_info_regression):
      for k in range(5, 25):

        # Pipeline con el selector y preprocesador 
        selector = make_pipeline(processor,
                              SelectKBest(m, k=k))

        # Entrenar selector
        selector.fit(X_train, y_train)

        # Asignar lista con variables escogidas con cada kbest al valor correspondiente en el diccionario
        if m == f_regression:
          # Asignar listas a la llave f_regression
          vars["f_regression"] += selector.get_feature_names_out().tolist()
        else:
          # Asignar listas a la llave mutual_info
          vars["mutual_info"] += selector.get_feature_names_out().tolist()

    # Crear data frame con los index como las variables y los valores de veces escogidas por cada metrica
    vars_kb = pd.DataFrame({i:pd.Series(j).value_counts() for i,j in vars.items()})
    vars_kb.index = vars_kb.index.str[5:]

    # Plot the DataFrame con variables y las veces escogidas
    if plot:
        # Ordenar DataFrame segun metrica escogida
        vars_kb.sort_values(by=plot_metric, ascending=True).plot(kind="barh")
        # Ubicar legenda
        plt.legend(loc=[0.7, 0.2])
        plt.title("Feature Importance (f_regression vs mutual information)")
        plt.show()

    # Listas con metricas
    metric = ["f_regression", "mutual_info"]
    criterion = ["squared_error", "absolute_error"]

    # DataFrame para guardar resultados de RFE
    vars_rfe = pd.DataFrame()

    # Recorrer cada metrica
    for m in metric:
      
      # Variable objetivo
      target = "Costo_por_evento"
      # Tomar las variables seleccionadas para el Kbest
      X = df_vars[vars_kb[m].sort_values(ascending=False).index.values.tolist()[:num_feat_kbest]]

      # Vector de variable objetivo convertido a horas
      y = df_vars[target]
      # Division de datos de evaluacion y entrenamiento
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      
      # Preprocesador de variables categoricas
      cat_processor = OrdinalEncoder()
      # Preprocesador de variables numericas
      num_processor = MinMaxScaler()

      # DataFrame de variables numericas  
      num_vals = X._get_numeric_data().columns.tolist()
      # DataFrame de variables categoricas
      cat_vals = X.select_dtypes("object").columns.tolist()
      
      # Union de procesadores
      processor = ColumnTransformer(transformers=[("cat", cat_processor, cat_vals), ("num", num_processor, num_vals)])

      for c in criterion:

        # Pipeline con el selector y preprocesador 
        selector = make_pipeline(processor,
                                RFE(DecisionTreeRegressor(criterion=c, random_state=42), n_features_to_select=num_rfe))

        # Entrenar selector
        selector.fit(X_train, y_train)

        # Guardar las variables seleccionadas con cada combianacion de criterios en las columnas    
        vars_rfe[c+f"_{m}"] = X.columns[selector.named_steps["rfe"].support_].values

    # Instanciar Counter
    counter = Counter()

    # Contar las variables mas repetidas en todos los criterios
    for i in vars_rfe.columns:
      counter.update(vars_rfe[i])

    # DataFrame con los nombres de las variables y las veces que aparecieron en cada criterio
    n_select = pd.DataFrame(counter.values(), index=counter.keys()).rename(columns={0:"count"}).sort_values(by="count", ascending=True)

    
    if plot:

        # Diagrama de barras de las variables mas recurrentes en cada criterio    
        n_select.plot(kind="barh", title="Numero de apariciones en los criterios", legend=False)
        plt.show() 

    # Diccionario con los nombres de combinaciones de criterios como columnas
    info = {i:[] for i in vars_rfe.columns}
    # Algoritmos a evaluar
    clfs = ["rf", "gb", "svr"]


    # DataFrame a usar en el entrenamiento de los modeloss
    df_train = df_vars.copy()

    # Iterar sobre cada combinación de variables seleccionadas y clasificadores
    for v in vars_rfe:
        for clf in clfs:

            target = "Costo_por_evento"
    
            # Extraer variables seleccionadas y variable objetivo
            vars = vars_rfe[v].values.tolist()
            target = "Costo_por_evento"
            
            # Extraer variables en X y variable objetivo en horas
            X = df_train[vars]
            y = df_train[target]
            
            # Dividir los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Entrenar el modelo y hacer predicciones
            model = fit_model(df_train, vars=vars, clf=clf)
            pred = model.predict(X_test)
            
            # Calcular el error absoluto medio y agregarlo al diccionario de info
            info[v].append(mean_absolute_error(y_test, pred))

    # Diccionario para almacenar el error absoluto medio para la combinación de variables mas recurrentes
    _ = {"recurrente": []}

    # Iterar sobre cada clasificador para la combinación de variables  mas presentes en todos los criterios
    for clf in clfs:
        # Extraer variables seleccionadas
        vars = n_select.index.values.tolist()
        
        # Extraer variables en X y variable objetivo en horas   
        X = df_train[vars]
        y = df_train[target]
        
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar el modelo y hacer predicciones
        model = fit_model(df_train, vars=vars, clf=clf)
        pred = model.predict(X_test)
        
        # Calcular el error absoluto medio y agregarlo al diccionario de info
        _["recurrente"].append(mean_absolute_error(y_test, pred))

    # Actualizar el diccionario de info con el error absoluto medio para la combinación 'recurrente'
    info.update(_)

    # Diccionario para almacenar el error absoluto medio usando todas las varaibles
    every = {"all_feat": []}

    # Iterar sobre cada clasificador usando todas las varaibles elegidas antes de algoritmos de seleccion
    for clf in clfs:
        
        # Extraer variables en X y variable objetivo en horas
        X = df_train.drop(to_drop, axis=1)
        y = df_train[target]
        
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar el modelo y hacer predicciones
        model = fit_model(df_train, vars=vars, clf=clf)
        pred = model.predict(X_test)
        
        # Calcular el error absoluto medio y agregarlo al diccionario 'every'
        every["all_feat"].append(mean_absolute_error(y_test, pred))

    # Actualizar el diccionario de info con el error absoluto medio para la combinación 'all_feat'
    info.update(every)

    # Crear DataFrame con la información del error absoluto medio
    eval_df = pd.DataFrame(info, index=[clfs])

    # Verificar si el directorio 'features' existe, si no, crearlo
    if os.path.exists('features'):
        pass
    else:
        os.mkdir("features")

    # Lista de DataFrames para almacenar
    to_store = [vars_kb, vars_rfe, n_select, eval_df]

    # Iterar sobre los DataFrame para almacenar y guardarlos como archivos pickle
    for i in to_store:
        for name, value in locals().items():
            if i is value:
                if name == "i":
                    pass
                else:
                    with open(f"features/{name}.pkl", "wb") as f:
                        pickle.dump(i, f)