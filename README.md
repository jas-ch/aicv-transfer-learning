# ASL-translator: American Sign Lanugage Translator

**ASL-translator** is an AI model that helps translating hand signs into the ASL alphabet for bridging communication when one doesn't know sign language.   The model reaches 99.9% accuracy and is easy to plug into any SDK or application for infinite possibilites. The model utilizes **ConvNeXt** and, after training, utilizes checkpoints for easy access to a well-trained save state. The model is trained with **ASL Alphabet Dataset** from Kaggle ([link here](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)). To run the inferencing, simply put picture files into the folder of asl/asl_alphbet_test/ and then run python ./classify.py -h for parameters usage. To run inferencing in a live stream mode, import ASL-translator as a module and invoke the function accordingly.  Sample output in output.txt file.


