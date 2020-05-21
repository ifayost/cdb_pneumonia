# cdb_pneumonia
Trabajo para la asignatura "Ciencia de Datos en Biomedicina" sobre la clasificación de radiografías con pneumonia.

Se ha entrenado un modelo que clasifica radiografias en las clases normal (sano), neumonía bacteriana y neumonía vírica. 
Se han aplicado tecnicas de Tranfer Learning, en particular, la arquitectura VGG16 preentrenada en el dataset de Image Net ha sido adaptada y reentrenada para clasificar radiografias en estas 3 clases.
Se ha utilizado para reentrenar la red el dataset [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) de Kaggle.

Ejemplo de clasificación:
![alt-text](https://github.com/ifayost/cdb_pneumonia/blob/master/example.jpeg "example")

## Dpendencias
* Python 3.7
* Pytorch 3.5
* Numpy 1.18.1
* Matplotlib 3.1.3
* Tqdm 4.46
* Captum 0.2

Puedes también crear un environment de conda con todas las dependencias necesarias para este repositorio con:
```
conda env create -f biomedicina.yml
``` 

## Instrucciones de uso

Para descargar los pesos de las redes ya entrenadas ejecutar el script download_wights.py:
```
python download_wights.py
```

A continación copia las radiografias que quieras diagnosticar en la carpeta test y ejecuta el script diagnose.py:
```
python diagnose.py
```

Si se quiere recortar las imagenes utilizando el método descrito en el **_informe.pdf_** se pude añadir el argumento `crop`:
```
python diagnose.py crop
```
de esta forma la radiografias serán recortadas antes de ser enviadas al modelo (¡Cuidado! Al aplicar el crop, las imagenes originales se sobreescriben por las imagenes recortadas en la carpeta test).

Las radiografias diagnosticadas por el modelo se guardan en la carpeta diagnoses con una imagen para cada radiografia diagnosticada en la que se incluye la radiaografia original, un mapa de calor con las zonas mas importantes para el diagnostico y las probabilidades de que la radiografia pertenzca a cada clase.

## Colab Notebooks

Todos los resultados descritos en el **_informe.pdf_** así como los pesos de los modelos entrenados se han obtenido utilizando la plataforma de Google Colab. Proporcionamos estos notebooks en la carpeta ColabNotebooks.