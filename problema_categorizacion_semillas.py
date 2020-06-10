#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
import warnings
import itertools

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVR 
from sklearn.feature_selection import RFECV 


# <h1> Cargando datos </h1>
# 

# In[2]:


semilla = pd.read_csv("seeds_dataset.csv")
#Informacion de los datos
print(semilla.info())


# <h1> Visualizando la distribución de clases</h1>
# 

# In[3]:


#Histograma del atributo clase
ax=plt.subplots(1,1,figsize=(10,8))
sns.countplot('Tipo',data=semilla)
plt.title("Tipos de semillas cuenta")
plt.show()


# <h1>Visualizando los histogramas de los atributos </h1>

# In[4]:


#Histograma de atributos predictores

semilla.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(15,15)
plt.show()


# <h1>Diagrama de cajas de los atributos o variables independientes.</h1>

# In[5]:


#boxplot de las variables numericas
semilla = semilla.drop('Id',axis=1)
box_data = semilla #variable representing the data array
box_target = semilla.Tipo #variable representing the labels array
sns.boxplot(data = box_data,width=1)
# sns.set(rc={'figure.figsize':(2,15)})
plt.show()


# Observando la correlación entre variables permite descubrir posibles dependencias entre las variables independientes.

# In[6]:


X = semilla.iloc[:, 0:7]
f, ax = plt.subplots(figsize=(10, 8))
corr = X.corr()
print(corr)
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
          cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax, linewidths=.5)
plt.show() 


# En la matriz de correlación se observa un alto coeficiente para las variables AreaA y PerimetroP. Podemos mirar el comportamiento de las dos variables utilizando regresión lineal.

# In[7]:


#observando relaciones entre los datos
sns.regplot(x='Area_A', y='Perimetro_P', data=semilla);
sns.set(rc={'figure.figsize':(20,5)})
plt.show()


# Podemos observar, con la gráfica anterior el comportamiento casi lineal que tienen estas dos variables, por lo que podemos contemplar eliminar una de ambas para el entrenamiento y predicción del modelo, pero tendremos que realizar más análisis.

# In[8]:


semilla = semilla.drop('Perimetro_P',axis=1)


# In[9]:


semilla


# Una vez observado y analizado las variables del conjunto de datos vamos a hacer una primera prueba preliminar para observar cómo se comportaría el modelo de red neuronal. La configuración de este primer modelo se indica a través de los parámetros de MPLClassifier
# 

# In[10]:


#Separando los datos en conjuntos de entrenaimiento y prueba
X = semilla.iloc[:, :-1].values
y = semilla.iloc[:, 6].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#Como esta es una primera prueba prelimintar coloco esta instrucción para que nos me saque un warning
#debido a que el modelo no alcanza a converger
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

#Entrenando un modelo de red neuronal MLP para clasificación
#MLPClassifier permite configurar las capas ocultas del modelo, la instrucción de abajo indica que el modelo tendrá
#dos capas ocultas cada una con 3 neuronas. Algo como esto hidden_layer_sizes = (3,3,2) indicarían tres capas ocultas con
#3,3 y 2 neuronas respectivamente
model =  MLPClassifier(hidden_layer_sizes = (3,3,2), alpha=0.01, max_iter=1000) 
model.fit(X_train, y_train) #Training the model


# Una vez entrenado el modelo, debemos evaluarlo sobre el conjunto de datos reservado para prueba, y utilizar algunas métricas para observar que tan bien quedo entrenado el modelo. En esta primera prueba utilizamos como métricas el porcentaje de precisión del modelo y la matriz de confusión.

# In[11]:


#Test the model
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictions))


# <h1>Busqueda de grilla </h1>

# Ahora vamos a ajustar los parámetros del modelo utilizando GridSearch
# 
# 

# In[12]:


parameters = {'activation': ['tanh','relu'],
              'solver': ['lbfgs','adam'], 
              'max_iter': [100,300,500], 
              'alpha': 10.0 ** -np.arange(3, 10), 
              'hidden_layer_sizes': [(3,3,2),(2,3,4), (3,2,2)],
              'random_state':[0,8,9]}


# In[13]:


mlp1 = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1, cv=5)
mlp1.fit(X_train, y_train)
print(mlp1.score(X_train, y_train))
print(mlp1.best_params_) 


# In[14]:


print(mlp1.best_params_["activation"])


# <h1>Corrección del modelo</h1>

# In[15]:


model =  MLPClassifier(solver = mlp1.best_params_["solver"], activation = mlp1.best_params_["activation"],  max_iter=mlp1.best_params_["max_iter"], random_state = mlp1.best_params_["random_state"] , alpha= mlp1.best_params_["alpha"], hidden_layer_sizes = mlp1.best_params_["hidden_layer_sizes"]) 
model.fit(X_train, y_train) #Training the model


# Haciendo un test del modelo corregido

# In[16]:


# #Test the model
predictions1 = model.predict(X_test)
print(accuracy_score(y_test, predictions1))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictions1))


# In[17]:


from sklearn.model_selection import KFold
kf=KFold(n_splits=2, shuffle=True, random_state = 2)

df = pd.DataFrame(semilla)

for valores_x, valores_y in kf.split(X):
    print('Entrenamiento: ', df.iloc[valores_x], 'Prueba:',  df.iloc[valores_y] )


# In[18]:


class_names = semilla['Tipo'].unique()
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
def CalcularMatrizConfusion(y_test, y_pred):
       
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

    plt.show()


# <h1>Matríz de confusión</h1>

# In[19]:


CalcularMatrizConfusion(y_test, predictions)


# In[ ]:




