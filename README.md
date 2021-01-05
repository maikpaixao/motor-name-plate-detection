# Tecvolt

## Modelos
Baixar os modelos abaixo e setar as variáveis de ambiente com o path do modelo.

* [Segmentação de placa](https://drive.google.com/file/d/12O9svb1C5FoAvRST9lg3FFEzpyP1GSk4/view?usp=sharing)
* [Segmentação de texto](https://drive.google.com/file/d/1LNbZ5WTGzqr9oa7Ciy9MhoO4ogUe0LLX/view?usp=sharing)

## Datasets
Os datasets utilizados para treinamento e avaliação dos modelos estão disponíveis nos links abaixo.
Os links contem apenas as imagens originais, sem o uso de data augmentation.

O treinamento do modelo de segmentação de placa foi realizado com a base de dados de [Treinamento](https://drive.google.com/file/d/1kFq3wIOKUgLriy-iXt3Xz8ifNNqbKDtO/view?usp=sharing). Foi aplicado técnicas de data augmentation na base de dados utilizando o scrip [augmentation.py](augmentation.py).

Para testar o modelo de segmentação de placa foi utilizado a base de dados de [Teste](https://drive.google.com/file/d/1hJYetwuD18dqmOSeG3o3v6qwQlMSH6Oc/view?usp=sharing). O modelo também foi testado na base de dados de teste corrigida, as imagens corrigidas da base de dados teste se encontram [aqui](https://drive.google.com/file/d/1OSjmlbtqPd8T8TxMzU2rITHPyXzkSPYj/view?usp=sharing). Para montar a base de teste corrigida é necessário baixar a base de [Teste](https://drive.google.com/file/d/1hJYetwuD18dqmOSeG3o3v6qwQlMSH6Oc/view?usp=sharing) e substituir nela as [imagens corrigidas](https://drive.google.com/file/d/1OSjmlbtqPd8T8TxMzU2rITHPyXzkSPYj/view?usp=sharing). 

Links para bases de dados:
* [Treinamento](https://drive.google.com/file/d/1kFq3wIOKUgLriy-iXt3Xz8ifNNqbKDtO/view?usp=sharing)
* [Teste](https://drive.google.com/file/d/1hJYetwuD18dqmOSeG3o3v6qwQlMSH6Oc/view?usp=sharing)
* [Teste Corrigido](https://drive.google.com/file/d/1OSjmlbtqPd8T8TxMzU2rITHPyXzkSPYj/view?usp=sharing)

## Requerimentos

* Tesseract 4.0+
  ```
  sudo apt-get install tesseract-ocr
  ```
  Baixar os modelos de linguagem com melhor acurácia: [eng.traineddata](https://github.com/tesseract-ocr/tessdata_best/blob/master/eng.traineddata) e [por.traineddata](https://github.com/tesseract-ocr/tessdata_best/blob/master/por.traineddata).
  
    Copiar modelos treinados para o diretório do tesseract: /usr/share/tesseract-ocr/4.00/tessdata/
* [requiriments.yaml](requiriments.yaml)
* [requiriments.txt](requiriments.txt)
