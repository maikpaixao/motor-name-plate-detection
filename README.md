# Motor Name Plate Detection using Computer Vision

## Models
Download the models below and set the environment variables with the model path.

* [Board segmentation](https://drive.google.com/file/d/12O9svb1C5FoAvRST9lg3FFEzpyP1GSk4/view?usp=sharing)
* [Text segmentation](https://drive.google.com/file/d/1LNbZ5WTGzqr9oa7Ciy9MhoO4ogUe0LLX/view?usp=sharing)

## Datasets
The datasets used for training and evaluating the models are available at the links below.
The links only contain the original images, without the use of data augmentation.

The training of the plate segmentation model was carried out with the [Training](https://drive.google.com/file/d/1kFq3wIOKUgLriy-iXt3Xz8ifNNqbKDtO/view?usp=sharing) database. Data augmentation techniques were applied to the database using the script [augmentation.py](augmentation.py).

To test the plaque segmentation model, the [Test](https://drive.google.com/file/d/1hJYetwuD18dqmOSeG3o3v6qwQlMSH6Oc/view?usp=sharing) database was used. The model was also tested on the corrected test database, the corrected images from the test database can be found [here](https://drive.google.com/file/d/1OSjmlbtqPd8T8TxMzU2rITHPyXzkSPYj/view?usp=sharing). To assemble the corrected test base it is necessary to download the [Test](https://drive.google.com/file/d/1hJYetwuD18dqmOSeG3o3v6qwQlMSH6Oc/view?usp=sharing) base and replace the [corrected images](https: //drive.google.com/file/d/1OSjmlbtqPd8T8TxMzU2rITHPyXzkSPYj/view?usp=sharing).

Links to databases:
* [Training](https://drive.google.com/file/d/1kFq3wIOKUgLriy-iXt3Xz8ifNNqbKDtO/view?usp=sharing)
* [Test](https://drive.google.com/file/d/1hJYetwuD18dqmOSeG3o3v6qwQlMSH6Oc/view?usp=sharing)
* [Fixed Test](https://drive.google.com/file/d/1OSjmlbtqPd8T8TxMzU2rITHPyXzkSPYj/view?usp=sharing)

## Requirements

* Tesseract 4.0+
   ```
   sudo apt-get install tesseract-ocr
   ```
   Download the language models with the best accuracy: [eng.traineddata](https://github.com/tesseract-ocr/tessdata_best/blob/master/eng.traineddata) and [por.traineddata](https://github. com/tesseract-ocr/tessdata_best/blob/master/por.traineddata).
  
     Copy trained models to the tesseract directory: /usr/share/tesseract-ocr/4.00/tessdata/
* [requiriments.yaml](requiriments.yaml)
* [requiriments.txt](requiriments.txt)
