<<<<<<< HEAD
Tongji's wood damge prediction project intending to predict different types of wood damage under different load or situations.

#Files area:
1.differential_model.py----which implements Foschi's and Yao's model
2.train_model.py----which utilizes Pytorch calling differential_model.py to get data, and to train/validate a network predicting wood internal damage. Test is also within this code by calling '--epoch=-1', and will given stastical result comparing to classical Foschi model.

#Operations to get result:
=======
# wood_prediction

Tongji's wood damge prediction project intending to predict different types of wood damage under different load or situations.
整体思路：
考虑不同木材性质、不同负载情况，从校准后的传统数学模型中采样，记录的数据不光包含其破坏时间曲线，更包含每一个样本在整个实验过程中的损伤过程。把这些损伤过程离散打乱，尝试使用神经网络来学习其中的损伤关系。


Files are:

1.differential_model.py
----which implements Foschi's and Yao's model， train_model.py will need this to get learning data. 

2.train_model.py
----which utilizes Pytorch calling differential_model.py to get data, and to train/validate a network predicting wood internal damage. Test is also within this code by calling '--epoch=-1', and will given stastical results comparing to classical Foschi model.

Operations to get result:
Opertaions and result images are:

---恒定荷载情况---：

程序接口说明：（单体木材的损伤过程在长时间实验中默认不显示，如要观察中间过程，可以设置Test_years_to_run<0.1）

实验1-1

python train_model.py --N=1000 --num_of_forces=2 --wood_types="All" --Years_to_run=50 --num_of_batch=50 --V --Test_years_to_run=50  --Alter_force_scaler=1.0

结果1-1：

python train_model.py --N=1000 --num_of_forces=2 --wood_types="All" --V --num_of_batch=50 --Years_to_run=50 --Test_years_to_run=50  --Alter_force_scaler=1.0 --Restart --epoch=-1 --Q

实验1-2

python train_model.py --N=1000 --num_of_forces=10 --wood_types="All" --V --num_of_batch=50 --Years_to_run=50 --Test_years_to_run=50  --Alter_force_scaler=1.0

结果1-2：

python train_model.py --N=1000 --num_of_forces=10 --wood_types="All" --V --num_of_batch=50 --Years_to_run=50 --Restart --Test_years_to_run=50  --Alter_force_scaler=1.0 --Restart --epoch=-1 --Q

---可变荷载情况---：

程序接口说明：（单体木材的损伤过程在长时间实验中默认不显示，如要观察中间过程，可以设置Test_years_to_run<0.1）

实验2-1：

python train_model.py --N=1000 --num_of_forces=10 --wood_types="All" --Years_to_run=50 --num_of_batch=500 --V --Test_years_to_run=50  --Alter_force_scaler=1.0

实验2-1（参考数据生成）：

python train_model.py --N=1000 --num_of_forces=10 --wood_types="All" --Years_to_run=50 --num_of_batch=500 --V --Test_years_to_run=50  --Alter_force_scaler=0.75

python train_model.py --N=1000 --num_of_forces=10 --wood_types="All" --Years_to_run=50 --num_of_batch=500 --V --Test_years_to_run=50  --Alter_force_scaler=1.25

结果2-1-1：

python train_model.py --N=1000 --num_of_forces=10 --wood_types="All" --Years_to_run=50 --num_of_batch=500 --V --Test_years_to_run=50  --Alter_force_scaler=0.75 --Restart --epoch=-1 --Q

结果2-1-2：

python train_model.py --N=1000 --num_of_forces=10 --wood_types="All" --Years_to_run=50 --num_of_batch=500 --V --Test_years_to_run=50  --Alter_force_scaler=1.25 --Restart --epoch=-1 --Q



实验2-2：

python train_model.py --N=1000 --num_of_forces=100 --wood_types="SPF_Q1" --Years_to_run=50 --num_of_batch=500 --V --Test_years_to_run=50  --Alter_force_scaler=1.0

实验2-1（参考数据生成）：

python train_model.py --N=1000 --num_of_forces=100 --wood_types="SPF_Q1" --Years_to_run=50 --num_of_batch=500 --V --Test_years_to_run=50  --Alter_force_scaler=0.75

python train_model.py --N=1000 --num_of_forces=100 --wood_types="SPF_Q1" --Years_to_run=50 --num_of_batch=500 --V --Test_years_to_run=50  --Alter_force_scaler=1.25

结果2-2-1：

python train_model.py --N=1000 --num_of_forces=100 --wood_types="SPF_Q1" --Years_to_run=50 --num_of_batch=500 --V --Test_years_to_run=50  --Alter_force_scaler=1.0 --Restart --epoch=-1 --Q

结果2-2-2：

python train_model.py --N=1000 --num_of_forces=100 --wood_types="SPF_Q1" --Years_to_run=50 --num_of_batch=500 --V --Test_years_to_run=50  --Alter_force_scaler=0.75 --Restart --epoch=-1 --Q

结果2-2-3：

python train_model.py --N=1000 --num_of_forces=100 --wood_types="SPF_Q1" --Years_to_run=50 --num_of_batch=500 --V --Test_years_to_run=50  --Alter_force_scaler=1.25 --Restart --epoch=-1 --Q

>>>>>>> 03c656c1cd2ae1f77d5bbada24e51a23be3a59cf
