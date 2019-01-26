# wood_prediction

Tongji's wood damge prediction project intending to predict different types of wood damage under different load or situations.

Files are:
1.differential_model.py----which implements Foschi's and Yao's model
2.train_model.py----which utilizes Pytorch calling differential_model.py to get data, and to train/validate a network predicting wood internal damage. Test is also within this code by calling '--epoch=-1', and will given stastical results comparing to classical Foschi model.

#Operations to get result:
