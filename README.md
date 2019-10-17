# Preparacion_de_datos_y_regresion
 
Diseñar y desarrollar un modelo de preparación de datos y otro de regresión que implemente un
descenso de gradiente capaz de resolver satisfactoriamente problemas de regresión lineal.
Estos modelos deben ser de propósito general y gestionar correctamente datos de entrada de
cualquier dimensión y características. Por tanto, el algoritmo de descenso de gradiente aceptará
como datos de entrada arrays bidimensionales (�, �), donde � representa el número de tuplas a
evaluar y � el número variables para cada tupla. Los datos de salida generados se corresponderán
con un vector unidimensional (�), donde � representa el número de tuplas evaluados.
Se incluirá una prueba de concepto para resolver el problema de regresión diabetes
(sklearn.datasests.load_diabetes). Se deberá configurar (mediante argumentos) los modelos
desarrollados para que los resultados obtenidos por el descenso de gradiente sean óptimos.
Para el desarrollo de estos modelos se hará uso del entorno de programación python y las librearía
numpy de python. Salvo para la importación de datasets, la librería sklearn no podrá ser utilizada
