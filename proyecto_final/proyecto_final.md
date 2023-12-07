# Modelo de Prediccion de Partidas de League of Legends

---

***Luciano Masuelli***

***Facundo Gaviola***

***Facultad de Ingeniería, Universidad Nacional de Cuyo***


---
## Introducción


El propósito de este proyecto es anticipar el resultado de una partida en League of Legends, determinando si un equipo 
en particular alcanzará la victoria. Para lograrlo, se emplearon algoritmos de Inteligencia Artificial, utilizando datos 
relacionados con ambos equipos que participan en el enfrentamiento, así como características clave que definen el inicio 
de la partida.

Para dar un poco de contexto, League of Legends (LoL) es un popular videojuego de 
estrategia en tiempo real que ha alcanzado una inmensa popularidad desde su lanzamiento en 2009. En este juego, dos 
equipos de cinco jugadores se enfrentan en un campo de batalla virtual, cada uno controlando un campeón con habilidades 
únicas. El juego ha dejado una marca indeleble en el mundo de los esports, destacándose por su enorme base de jugadores,
la celebración del Campeonato Mundial anual con premios millonarios, y el establecimiento de ligas profesionales en
diversas regiones.

Por todo lo mencionado anteriormente, se puede considerar de una gran importancia el poder predecir el resultado de una
partida de LoL, ya que esto puede ser utilizado, ya sea para apuestas deportivas, mejorar el rendimiento de un equipo o
simplemente comprender las distintas variables que se relacionan entre sí durante una partida a la hora de conseguir la
victoria.

Se implementaron modelos de inteligencia artificial, como Random Forest o alguna variante de boosting, con el objetivo 
de prever los resultados de las partidas en League of Legends.   
La elección de estas técnicas de inteligencia artificial se sustenta en varios factores cruciales:
* Complejidad del juego: League of Legends presenta una complejidad considerable, con numerosas variables que pueden 
ejercer una influencia significativa en el desarrollo y desenlace de cada partida.
* Gran cantidad de datos: Debido a la popularidad de LoL, es una tarea facil la recopilación de una gran cantidad de 
partidas para poder entrenar los modelos.
* Analisis de variables: Mediante la aplicacion de tecnicas de inteligencia artificial, se puede analizar cuales son las
variables que mas influyen en el resultado de una partida, y cuales son las que menos influyen. Algortimos como Random
Forest permiten analizar la importancia de cada variable a la hora de realizar una prediccion.

Para la realizacion del proyecto se utilizaran partidas de la region de Corea de la temporada 2020 de LoL. Estas 
partidas fueron obtenidas de la pagina web de Kaggle, y se pueden encontrar en el siguiente link:
https://www.kaggle.com/datasets/gyejr95/league-of-legendslol-ranked-games-2020-ver1/data?select=match_winner_data_version1.csv

Se intentarán utilizar como variables predictoras a aquellos datos que puedan ser obtenidos al comienzo de una partida 
o en su defecto, en los primeros minutos de transcurrida la misma. Esto se debe a que se busca realizar una predicción
lo más temprana posible en la partida.

---

## Marco teórico

**Árboles de Decision**

Los árboles de decision representan una función que toma como entrada un vector de atributos y valores y devuelve como 
resultado una decision. Cada nodo interno del arbol representa una prueba sobre un atributo, cada rama representa el
resultado de una prueba y cada nodo hoja representa una clase. El camino desde la raíz hasta una hoja representa una
regla de clasificación.

El algoritmo en pseudocodigo utilizado para la creacion de un arbol de decision es el siguiente:
* plurality-value devuelve el valor mas comun entre los ejemplos.
* A es el atributo que mejor clasifica los ejemplos.
* IMPORTANCE es una funcion que mide la importancia de un atributo.
* v_k es el valor de A que corresponde a la rama del arbol.
* e.A es el valor del atributo A en el ejemplo e.
```
function decision-tree-learning(examples, attributes, parent_examples) returns a tree
    if examples is empty then return plurality-value(parent_examples)
    else if all examples have the same classification then return the classification
    else if attributes is empty then return plurality-value(examples)
    else
        A <- argmax a in attributes IMPORTANCE(a, examples)
        tree <- a new decision tree with root test A
        for each value v_k of A do
            exs <- {e : e in examples and e.A = v_k}
            subtree <- decision-tree-learning(exs, attributes - A, examples)
            add a branch to tree with label (A = v_k) and subtree subtree
        return tree
```

**Random Forest**

Random Forest es un algoritmo de aprendizaje basado en arboles de decision. El algoritmo crea una serie de arboles de
decision, de la misma manera que lo hace un algoritmo de bagging, pero en este caso, en el momento de en que cada arbol
del modelo va a crear nuevas ramas, se selecciona un subconjunto aleatorio de `m` variables predictoras del total de 
variables predictoras `p`. En cada creacion de ramas se seleccionan `m` variables predictoras distintas. Esto se hace para
evitar que los arboles esten altamente correlacionados entre si y asi tener un modelo mas confiable. Generalmente se
utiliza `m = sqrt(p)` para la cantidad de variables seleccionadas.

**Boosting**

Boosting es un algoritmo de aprendizaje que, al igual que Random Forest, se basa en arboles de decision para realizar 
regresiones o clasificaciones. En boosting no se involucran tecnicas de boostraping, sino que se utilizan distintas
versiones modificadas de los datos de entrenamiento. El algoritmo combina multiples arboles de decision, ˆf 1,..., ˆf B
para crear un modelo de prediccion f(x) = sum(b=1 to B) f_b(x). Es importante destacar que los arboles de decision son
pequeños, es decir, tienen pocos niveles y pocos nodos. Esto mejora lentamente el rendimiento de f(x). El pseudocodigo
del algoritmo es el siguiente:
* ri es el residuo de la prediccion en el ejemplo i.
* lambda es un parametro de contraccion.
* d es la cantidad de divisiones que se realizan en cada arbol
* fb es el arbol de decision b
* f es el modelo de prediccion

```
1. Initialize f_0(x) = 0
2. For b = 1 to B:
    (a) Fit a tree ˆf b with d splits (d + 1 terminal nodes) to the training data (X, r).
    (b) Update ˆf by adding in a shrunken version of the new tree: ˆf(x) ← ˆf(x) + λ ˆfb(x).
    (c) Update the residuals,ri ← ri − λ ˆfb(xi).
3. Output the boosted model, f(x) = sum(b=1 to B) λf_b(x).
```

---


## Diseño Experimental
 
### Limpieza de datos
El primer paso para la realizacion del proyecto es la adecuacion del dataset para poder ser utilizado correctamente por
los algoritmos de inteligencia artificial. Para esto se realizo una limpieza de los datos, eliminando aquellos que no 
sean necesarios para las predicciones de los modelos, tales como, datos de temporada, creacion de partida, modo de juego
(solo nos enfocamos en el modo de juego classic) y estadisticas de los jugadores al finalizar la partida.

Tambien se modifico la presentacion de los datos, ya que estos se encontraban almacenados en estructuras de datos, tales
como listas o diccionarios, dificultando su uso. Para ello se crearon nuevas columnas en el dataset, donde se guardaran
los datos de dichas estructuras.

Finalmente se transformaron los datos categoricos en datos numericos, como por ejemplo, transformar los valores True y 
False en 1 y 0 respectivamente, o transformar los nombres de los campeones en numeros enteros. Esto se realizo con el 
fin de que los algoritmos puedan trabajar de forma correcta con los datos.

La principal metrica que se utilizará para la evaluación del modelo será la cantidad de partidas predichas correctamente
sobre la cantidad total de partidas. sobre esta metrica se calculará la precision, sensibilidad, exactitud y F-score.

### Random Forest

Para la aplicación de este modelo se utilizó la libreria de `scikit-learn`. Tambien se hizo uso de las librerias `pandas`
y `numpy` para la manipulacion de datos y `matplotlib` y `seaborn` para realizar ciertos graficos de los datos y 
resultados.

Antes de realizar el modelo de prediccion, se analizo el balanceo de clases del dataset. Se observó que dentro del mismo
habia una cantidad similar de partidas ganadas como de partidas perdidadas, por lo que no fue necesario el uso de 
tecnicas como oversampling, undersampling o SMOTE para balancear las clases. En la imagen se muestra como se distriyen
las clases dentro del conjunto de entrenamiento y el conjunto de prueba.

![](C:\Users\Facu\PycharmProjects\Proyecto_Final_IA\proyecto_final\images\random_forest\RF_balance_clases.png)

Al utilizar el algoritmo Random Forest se notó que la precision del modelo se encontraba entre un 70% y 75%
dependiendo del valor de 'Random_state' utilizado, tanto cuando se evaluaban utilizando el conjunto de prueba como el 
conjunto de entrenamiento. Se vio que, en general, al modificar los parametros de random forest tales como la cantidad 
de arboles, la cantidad de ejemplos minima para la creacion de una nueva rama, o los criterios de ramificacion, no se 
obtienen mejoras significativas en las metricas del modelo.

Al igual que la metrica de precision, tanto la sensibilidad, exactitud y F-score se encontraban entre un 70% y 75%

Las siguientes imagenes muestran las matrices de confusion y metricas del modelo con un random_state = 42.

**Conjunto de Prueba**

![](C:\Users\Facu\PycharmProjects\Proyecto_Final_IA\proyecto_final\images\random_forest\RF_matriz_confusion_test.png)

![](C:\Users\Facu\PycharmProjects\Proyecto_Final_IA\proyecto_final\images\random_forest\RF_metricas_test.png)

**Conjunto de Entrenamiento**

![](C:\Users\Facu\PycharmProjects\Proyecto_Final_IA\proyecto_final\images\random_forest\RF_matriz_confusion_train.png)

![](C:\Users\Facu\PycharmProjects\Proyecto_Final_IA\proyecto_final\images\random_forest\RF_metricas_train.png)


También se realizó un análisis de la importancia de las distintas variables predictoras a la hora de realizar el modelo
de prediccion con random forest. En las imagenes se puede observar que la variable mas influyentes es 'firstTower', 
mientras que las variables 'ban1', 'ban2', 'champ1', 'champ2', etc no realizan un aporte significativo al modelo.

![](C:\Users\Facu\PycharmProjects\Proyecto_Final_IA\proyecto_final\images\random_forest\RF_importancia_columnas.png)

![](C:\Users\Facu\PycharmProjects\Proyecto_Final_IA\proyecto_final\images\random_forest\RF_importancia_columnas_grafico.png)

### Gradient Boosting

Se aplicó también el algoritmo de Gradient Boosting para la realización del modelo de predicción. Este es una variante 
del algoritmo de Boosting en el que se van creando secuencialmente distintos modelos simples de árboles de decision, donde 
en cada iteración se intenta corregir los errores del modelo anterior. De esta forma, se va creando un modelo de 
predicción más robusto y preciso. Para la creación del modelo se hizo uso de la libreria `xgboost`. Al igual que en el 
caso de Random Forest, se utilizó la libreria `pandas` para la manipulación de datos. 

A la hora de realizar el modelo predictivo de clasificacion, se encontro con que el la precision del mismo se encontraba
entre alrededor del 73% cuando se evaluaba sobre el conjunto de prueba y un 83% cuando se evaluaba sobre el conjunto de
entrenamiento. A diferencia de random forest, no se observa una variacion dependiendo del valor de random_state. sin 
embargo, al igual que en random forest, a la hora de modificar los parametros del modelo, no se observan mejoras en el 
conjunto de prueba, pero si en el conjunto de entrenamiento en donde se llego a un 88%, lo que indica que lo unico que
logramos modificando dichos parametros es un sobreajuste del modelo.

Las metricas de sensibilidad, exactitud y F-score se encontraban entre un 73% en el conjunto de prueba y un 83% en el
conjunto de entrenamiento.

Las siguientes imagenes muestran las matrices de confusion y metricas del modelo

**Conjunto de Prueba**

![](C:\Users\Facu\PycharmProjects\Proyecto_Final_IA\proyecto_final\images\boosting\B_matriz_y_metricas_test.png)

**Conjunto de Entrenamiento**

![](C:\Users\Facu\PycharmProjects\Proyecto_Final_IA\proyecto_final\images\boosting\B_matriz_y_metricas_train.png)

Al analizar las variables mas importantes con el modelo de boosting nos encontramos con un escenario similar al modelo
de random forest en donde la variable mas importante es 'firstTower' y las variables 'ban' y 'champ' no realizan un 
gran aporte.

![](C:\Users\Facu\PycharmProjects\Proyecto_Final_IA\proyecto_final\images\boosting\B_importancia_columnas.png)




---

## Análisis y discusión de resultados


---

## Conclusiones finales


---

## Bibliografía

[1] Stuart Russell and Peter Norvig. Artificial Intelligence: A Modern Approach (3rd. ed.). Pearson, 2010.

[2] Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani.An Introduction to Statistical Learning with 
Applications in R. 

