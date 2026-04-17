[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Ne9Ma6s0)

model:
 - pytorch
 - uses ConvNeXt
 - saves checkpoints as .tar, eval (test) in classify.py

uses [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) from Kaggle
 - no manual edits made to dataset
 - splits train data 80%/20% for train + val
 - test data used as is, other/new data can be applied thru file path
