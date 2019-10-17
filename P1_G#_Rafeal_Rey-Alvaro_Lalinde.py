#AUTORES: Rafael Rey & Alvaro Lalinde

from sklearn import datasets
import numpy as np
import functools

# Metodo sacartuplas:
# Funcion: elimina las variables de X_train y X_test que no aporten demasiada informacion(varianza<0.5)
#          tambien elimina datos de varianza, media, mediana, IQR y desviacion típica de la variable.
# Parametros que recibe: X_train, X_test, varianza, media, mediana, IQR y desviacion_tipica
# Devuelve: X_train, X_test, varianza, media, mediana, IQR y desviacion_tipica modificados
def sacartuplas(X_train, X_test, varianza, media, mediana, IQR, desviacion_tipica):
    for tuplasI, tuplas in reversed(list(enumerate(varianza))):
        if tuplas<0.5:
            X_train = np.delete(X_train, tuplasI, axis=1)
            X_test = np.delete(X_test, tuplasI, axis=1)
            varianza=np.delete(varianza,tuplasI)
            mediana= np.delete(mediana, tuplasI)
            media= np.delete(media, tuplasI)
            IQR= np.delete(IQR, tuplasI)
            desviacion_tipica= np.delete(desviacion_tipica, tuplasI)
            print("Eliminamos variable(feature selection)")
    return X_train, X_test, varianza, media, mediana, IQR, desviacion_tipica

# Metodo sacartuplas:
# Funcion: elimina las columnas de X_train y X_test que no aporten demasiada informacion(varianza<0.5)
#          tambien elimina datos de la columna eliminada de varianza, media, mediana, IQR y desviacion típica
# Parametros que recibe: X_train, X_test, varianza, media, mediana, IQR y desviacion_tipica
# Devuelve: X_train, X_test, varianza, media, mediana, IQR y desviacion_tipica modificados
def sacartuplas(X_train, X_test, varianza, media, mediana, IQR, desviacion_tipica):
    for tuplasI, tuplas in reversed(list(enumerate(varianza))):
        if tuplas<0.5:
            X_train = np.delete(X_train, tuplasI, axis=1)
            X_test = np.delete(X_test, tuplasI, axis=1)
            varianza=np.delete(varianza,tuplasI)
            mediana= np.delete(mediana, tuplasI)
            media= np.delete(media, tuplasI)
            IQR= np.delete(IQR, tuplasI)
            desviacion_tipica= np.delete(desviacion_tipica, tuplasI)
    return X_train, X_test, varianza, media, mediana, IQR, desviacion_tipica


# Metodo normalizacionX:
# Funcion: Normaliza todos los datos del dataset que recibe
# Parametros que recibe: Dataset, array con los valores de la mediana, array con los valores del IQR,
#                        array con los valores de la media y array con los valores de la desviacion tipica.
# Devuelve: Dataset normalizado
def normalizacionX(datos,mediana,IQR,media, desviaciontipica):
    #Normalizados con mediana y IQR debido a la presencia de valores atipicos
    def normalizarConValoresAtipicos(x): return (x-mediana)/IQR
    #Normalizacion estandar, se realiza con la media y la desviacion típica
    def normalizarConMedia(x):return (x-media)/desviaciontipica
    datos= np.array(list(map(normalizarConValoresAtipicos, datos)))

    return(datos)

# Metodo calcularh:
# Funcion: Calcula las prediciones en funcion de las zetas recibidas
# Parametros que recibe: Lista de datos, array con los valores de las thetas y theta0
# Devuelve: predicion
def calcularh(X_train,th,th0):

    h=th0+X_train@th

    return h

# Metodo calcularzetas:
# Funcion: Calcula las nuevas thetas y theta0 para minimizar la funcion de error
# Parametros que recibe: Dataset de datos de entrada, dataser de datos de salida, alpha,
#                        lista de thetas y theta0
# Devuelve: nuevas thetas y theta0
def calcularthetas(X_train,y_train,alpha,th0,th,h):
    #Calaculo de zeta0

    w=((alpha/(len(X_train)))*(np.sum(h-y_train)))
    th0=th0-w

    #Derivada de la funcion de coste
    E=(alpha/(len(X_train)))*(((h-y_train)@X_train))
    #Calculo de la nuevas zetas
    th=th-E

    return th0, th
# Metodo eliminarRuido:
# Funcion: Elimina las tuplas en las que tengamos ruido  para una mejor aproximacion
# Parametros que recibe: Dataset de entradas, array de salidas, array con los valores
#  de la desviacion tipica, array con los valores de la media
# Devuelve: Dataset modificado
def eliminarRuido(datosX, datosY, IQR, mediana):

    ruidoeliminado=0
    for tuplasI, tuplas in reversed(list(enumerate(datosX))):
        valores_ruidosos=0
        for iteracionI, iteracion in enumerate(tuplas):
            max = mediana[iteracionI]+ IQR[iteracionI] *1.5
            min = mediana[iteracionI] - IQR[iteracionI] *1.5

            if iteracion>max or iteracion<min:
                valores_ruidosos=valores_ruidosos+1


        if valores_ruidosos >=1 :
            datosX=np.delete(datosX, tuplasI, axis=0)
            datosY=np.delete(datosY, tuplasI)
            ruidoeliminado=ruidoeliminado+1
    print(ruidoeliminado,"Tuplas de ruido eliminadas")


    return datosX, datosY

class Preprocessor(object):
    def __init__(self,**args):
        self.__dict__.update(args)
        #OPTIONAL TODO
        #Parametros que usamos para estudiar los datos y su preparacion
        self.mediana = 0
        self.desviacion_tipica = 0
        self.media = 0
        self.IQR = 0

    def fit(self, X, y=None):
        '''
        Recibe los datos de entrada X y opcionalmente los datos de salida esperados y para preparar los modelos necesarios para
        el preprocesado de datos.

        X   Matriz bidimiensional con forma (u,v) donde u es el número de tuplas y v el número de variables
        y   (opcional) Vector unidimensional con longitud u que contiene las salidas esperadas para cada una de las tuplas de X
        No devuelve nada
        '''
        #TODO
        self.mediana = np.median(X, axis = 0)
        self.media = np.mean(X,  axis = 0)
        self.desviacion_tipica = np.std(X,  axis = 0)
        self.varianza=np.var(X, axis=0)
        self.IQR = np.subtract(*np.percentile(X, [75, 25], axis = 0))

    def transform(self, X_train, X_test, y_train):
        '''
        Recibe los datos de entrada X y opcionalmente los datos de salida esperados y para preprocesar los datos de entrada.

        X   Matriz bidimiensional con forma (u,v) donde u es el número de tuplas y v el número de variables
        y   (opcional) Vector unidimensional con longitud u que contiene las salidas esperadas para cada una de las tuplas de X
        Devuelve el conjunto de datos X e y transformado
        '''
        #TODO
        X_train, y_train = eliminarRuido(X_train, y_train, self.IQR, self.mediana)

        X_train = normalizacionX(X_train,self.mediana,self.IQR,self.media,self.desviacion_tipica)
        X_test = normalizacionX(X_test,self.mediana,self.IQR,self.media,self.desviacion_tipica)

        return X_train, X_test, y_train


class GradientDescent(object):
    def __init__(self, alpha, **args):
        self.alpha = alpha
        self.__dict__.update(args)
        #OPTIONAL TODO
        self.theta0 = []
        self.thetas = []
        self.mse_ = 99999
        self.mseNew = 99998

    def fit(self, X_train, X_test, y_train, y_test):
        '''
        Recibe los datos de entrada X y los datos de salida esperados y para entrenar el modelo de descenso de gradiente.

        X   Matriz bidimiensional con forma (u,v) donde u es el número de tuplas y v el número de variables
        y   Vector unidimensional con longitud u que contiene las salidas esperadas para cada una de las tuplas de X
        No devuelve nada
        '''
        #TODO
        self.theta0=np.random.uniform(-1,1)
        self.thetas=[np.random.uniform(-1,1)]*(len(X_train.T))
        while self.mseNew<self.mse_:
            self.fit_step(X_train, y_train)
            self.mse_ = self.mseNew
            self.mseNew=self.mse(X_test, y_test)

    def fit_step(self, X, y):
        '''
        Recibe los datos de entrada X y los datos de salida esperados y para realizar una actualización en batch
        de los valores thetas

        X   Matriz bidimiensional con forma (u,v) donde u es el número de tuplas y v el número de variables
        y   Vector unidimensional con longitud u que contiene las salidas esperadas para cada una de las tuplas de X
        No devuelve nada
        '''
        #TODO
        h=calcularh(X_train,self.thetas,self.theta0)

        self.theta0, self.thetas =calcularthetas(X_train, y_train, self.alpha, self.theta0, self.thetas ,h)

    def predict(self, X):
        '''
        Recibe los datos de entrada X para estimar la salida estimada y_ segun el modelo entrenado

        X   Matriz bidimiensional con forma (u,v) donde u es el número de tuplas y v el número de variables
        y   Vector unidimensional con longitud u que contiene las salidas esperadas para cada una de las tuplas de X
        Devuelve un vector unidimensional y_ de longitud u el cual contiene el conjunto de salidas estimadas para X
        '''
        #TODO
        y_=np.zeros(len(X_test))
        #Calculamos las prediciones:
        y_=calcularh(X_test,self.thetas,self.theta0)

        return y_

    def mse(self, X, y):
        '''
        Recibe los datos de entrada X y los datos de salida esperados y para obtener el error cuadratico medio
        del modelo entrenado

        X   Matriz bidimiensional con forma (u,v) donde u es el número de tuplas y v el número de variables
        y   Vector unidimensional con longitud u que contiene las salidas esperadas para cada una de las tuplas de X
        Devuelve el error cuadratico medio (float)
        '''
        #TODO
        predicion=np.zeros(len(X_test))
        errores=np.zeros(len(X_test))
        #Llamamos a predic para calcular las prediciones con los datos de testeo
        predicion=self.predict(X_test)
        #Calculo de los errores de la predicion:
        errores=(predicion-y_test)
        #Calculo de error cuadratico medio:
        errorcuadraticom=(np.sum(np.square(errores))/(len(X_test)))

        return errorcuadraticom

if __name__ == "__main__":

    np.random.seed()
    X, y = datasets.load_diabetes(return_X_y=True)
     #Dividimos los datos en X_train, y_train, X_test y y_train
    msk = np.random.rand(len(X)) >= 0.25
    print("Tamaño del dataset",len(y))
    print("Tamaño del train",len(y[msk]))
    print("Tamaño del test",len(y[~msk]))
    X_train = X[msk]
    y_train = y[msk]

    X_test = X[~msk]
    y_test = y[~msk]

    # Preprocesado de datos (utilizar el modelo de preprocesado creado)
    #TODO
    modelo = Preprocessor()
    modelo.fit(X_train)

    print("Tamaño antes de transformar(tuplas,Variables):",len(X_train),", ",len(X_train.T))
    X_train, X_test, y_train=modelo.transform(X_train, X_test, y_train)
    print("Tamaño despues de transformar(tuplas,Variables):",len(X_train),", ",len(X_train.T))


    # Creación y entrenamiento del modelo de descenso de gradiente
    #TODO
    modelo = GradientDescent(0.02)
    modelo.fit(X_train, y_train, X_test, y_test)

    # Impresion del error cuadratico medio obtenido con el modelo
    #TODO
    print("ERROR:")
    print(modelo.mse(X_test, y_test))
