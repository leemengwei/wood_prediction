from __future__ import print_function
from IPython import embed
import pickle
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import differential_model
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import sys,os
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import copy
import collections
import pandas as pd
import matplotlib.colors as colors

class BaseModel(nn.Module):
    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell):
        super(BaseModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = layerNum
        if cell == "RNN":
            self.cell = nn.RNN(input_size=self.inputDim, hidden_size=self.hiddenNum,
                num_layers=self.layerNum, dropout=0.0, nonlinearity="tanh", batch_first=True,)
        if cell == "LSTM":
            self.cell = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenNum,
                num_layers=self.layerNum, dropout=0.0, batch_first=True, )
        if cell == "GRU":
            self.cell = nn.GRU(input_size=self.inputDim, hidden_size=self.hiddenNum,
                num_layers=self.layerNum, dropout=0.0, batch_first=True, )
        self.fc = nn.Linear(self.hiddenNum, self.outputDim)

class RNNNet(BaseModel):
    def __init__(self, inputDim, hiddenNum, layerNum, outputDim, cell):
        super(RNNNet, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell)
    def forward(self, x, batchSize):
        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        rnnOutput, hn = self.cell(x, h0) # rnnOutput 12,20,50 hn 1,20,50
        hn = hn.view(batchSize, self.hiddenNum)
        fcOutput = self.fc(hn)
        return fcOutput

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_depth, output_size, device):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.hidden_depth = hidden_depth
        self.fc2 = nn.Linear(hidden_size, output_size)  
        self.relu = nn.ReLU()
        self.sp = nn.Softplus()
        self.th = nn.Tanh()
        self.sg = nn.Sigmoid()
        self.fcns = nn.ModuleList()   #collections.OrderedDict()
        self.bns = nn.ModuleList()   #collections.OrderedDict()
        for i in range(self.hidden_depth):
            self.fcns.append(nn.Linear(hidden_size, hidden_size).to(device))
            self.bns.append(nn.BatchNorm1d(hidden_size).to(device))
    def forward(self, x):
        out = self.fc1(x)
        for i in range(self.hidden_depth):
            out = self.fcns[i](out)
            out = self.bns[i](out)
            out = self.relu(out)
        out = self.fc2(out)
#        out = self.sg(out)
        return out

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    LOSS = 0
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target) if args.lr<=1 else F.mse_loss(output, target)*args.lr
        loss.backward()
        LOSS += F.mse_loss(output, target, reduction='sum').item() # sum up batch loss
        optimizer.step()
        if (batch_idx % args.log_interval == 0 or batch_idx == len(train_loader)-1)and(batch_idx!=0):
            pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]. Loss: {:.8f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))
            pass
    train_loss_mean = LOSS/len(train_loader.dataset)
    print("Train Epoch: {} LOSS:{:.1f}, Average loss: {:.8f}".format(epoch, LOSS, train_loss_mean))
    return train_loss_mean

def validate(args, model, device, validate_loader):
    model.eval()
    LOSS = 0
    outputs_record = np.array([])
    targets_record = np.array([])
    with torch.no_grad():
        pbar = tqdm(validate_loader)
        for idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            LOSS += F.mse_loss(output, target, reduction='sum').item() # sum up batch loss
            outputs_record = np.append(outputs_record, output)
            targets_record = np.append(targets_record, target)
            pbar.set_description('Validate: [{}/{} ({:.0f}%)]'.format(idx*len(data), len(validate_loader.dataset), 100.*idx/len(validate_loader)))
    validate_loss_mean = LOSS/len(validate_loader.dataset)
    print('Validate set LOSS: {:.1f}, Average loss: {:.8f}'.format(LOSS, validate_loss_mean))
    return validate_loss_mean, outputs_record, targets_record

def test(args, model, device, datas, Preprocessor, canadian_t, canadian_curve, wood_type, stress_level, color_list):
    print("Running recursive loop...")
    datas = datas.to(device)
    #model.train()
    model.eval()
    end_t = int(24*365*args.Test_years_to_run)
    process_alphas_list = np.empty(shape=(0, datas.shape[0]))
    t = 0
    t_list_for_alphas = np.array([])
    t_list_for_cum = np.array([])
    cum_list = np.array([])
    previous_alpha = Preprocessor.just_unormalize_alpha(datas[:,Preprocessor.which_is_alpha]).reshape(-1,1)
    with torch.no_grad():
        while t<end_t:
            output = model(datas)
            t += args.Time_step
            damage = Preprocessor.just_unormalize_output(output)
            previous_alpha = damage + previous_alpha
            datas[:,Preprocessor.which_is_alpha] = Preprocessor.just_normalize_alpha(previous_alpha).view(-1)
            idx = np.where(previous_alpha>=1)[0]
            cum_list = np.append(cum_list, len(idx)/datas.shape[0])
            t_list_for_cum = np.append(t_list_for_cum, t)
            if (t%(24*365*10)==0):
                plt.clf()
                #Fig1:
                ax1 = plt.subplot(121)
                if args.Test_years_to_run>0.1:
                    plt.text(-0.01, -0.01, "Long-Term\nprediction\n\nSo Omit\nindividuals")
                else:
                    t_list_for_alphas = np.append(t_list_for_alphas, t)
                    process_alphas_list = np.vstack((process_alphas_list, previous_alpha.reshape(-1).cpu().numpy()))
                ax1.plot(np.log10(t_list_for_alphas), np.clip(process_alphas_list,0,1)[:,::int(process_alphas_list.shape[1]/1000)])
                #Fig2:
                ax2 = plt.subplot(122)
                ax2.plot(np.log10(canadian_t), canadian_curve, '--', label="Canadian Model of %s %s"%(wood_type, np.round(stress_level, 2)), color=color_list[wood_type+str(stress_level)])
                tmp_len = min(canadian_curve.shape[0], cum_list.shape[0])
                RR = np.round(np.corrcoef(canadian_curve[:tmp_len], cum_list[:tmp_len])[0,1]**2*100, 2)
                ax2.plot(np.log10(t_list_for_cum), cum_list, label="Network Result of %s %s, D:%s%%"%(wood_type, np.round(stress_level, 2), RR), color=color_list[wood_type+str(stress_level)])
                ax2.legend(loc="upper left", fontsize=8)
                plt.draw()
                plt.pause(0.001)
                #embed()
                print("Forward time: %s days, %s years, RR:%s%%"%(t/24, t/24/365, RR))
            if args.Alter_force_scaler!=1 and t==24*365*1:   #We alter force one year later
                print("Altering force to %s times."%args.Alter_force_scaler)
                physical_data = Preprocessor.unormalize_all(np.hstack((datas.cpu().numpy(),np.tile(0,(datas.shape[0],1)))))
                physical_data[:, Preprocessor.which_is_force] = physical_data[:, Preprocessor.which_is_force]*args.Alter_force_scaler
                datas = torch.FloatTensor(Preprocessor.normalize_all(physical_data)[:,:-1]).to(device)
                pass
    plt.close()
    return t_list_for_cum, cum_list, t_list_for_alphas, process_alphas_list

class data_preprocessing(object):
    def __init__(self, _data_):
        data = copy.deepcopy(_data_)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.which_is_force = 0
        self.which_is_alpha = int(np.array(list(set(list(np.where(data.max(axis=0)<=1)[0])) & set(list(np.where(data.min(axis=0)>=0)[0])))).min()) 
    def clean(self, _data_):
        data = copy.deepcopy(_data_)
        data = np.delete(data, np.where(data[:,self.which_is_alpha]==0), axis=0) #直接删除alpha等于零的所有点，因为要做log10......应该没有了， 都是1e-10
        data[np.where(data[:,-1]==0)]=np.hstack((data[np.where(data[:,-1]==0)][:,:-1], np.tile(data[np.where(data[:,-1]!=0)].min(),(data[np.where(data[:,-1]==0)].shape[0],1)))) #对于损伤等于零的点，赋值样本中的最小损伤，之后方便取log10. (赋值方式只能这么麻烦。)
        return data
    def get_stastics(self, _data_):
        data = copy.deepcopy(_data_)
        data[:,self.which_is_alpha] = np.log10(data[:,self.which_is_alpha])  #Subtle variable damage
        data[:,-1] = np.log10(data[:,-1])  #Subtle variable damage
        #self.mean = data.mean(axis=0)
        self.mean = np.array([24.86194164, -5.95217765, 45.41826586, -8.64803982])
        #self.std = data.std(axis=0)
        self.std = np.array([5.07882468, 3.00969408, 8.8454502 , 3.4259231 ])
        print(self.mean, self.std)
    def normalize_all(self, _data_):
        data = copy.deepcopy(_data_)
        data[:,self.which_is_alpha] = np.log10(data[:,self.which_is_alpha])  #Subtle variable alpha
        data[:,-1] = np.log10(data[:,-1])  #Subtle variable damage
        data = (data-self.mean)/self.std
        if np.isnan(data).any():
            print("Notice: There is NAN when normalizing input data...")
            print("This is either because that data series are all the same (All constant force? All no damage 0?) or there're just too few data (only one data)")
            sys.exit()
        return data
    def just_normalize_alpha(self, _data_):
        data = copy.deepcopy(_data_)
        data = np.log10(data)
        data = (data-self.mean[self.which_is_alpha])/self.std[self.which_is_alpha]
        return data
    def just_normalize_output(self, _data_):
        data = copy.deepcopy(_data_)
        data = np.log10(data)
        data = (data-self.mean[-1])/self.std[-1]
        return data

    def unormalize_all(self, _data_):
        data = copy.deepcopy(_data_)
        data = data*self.std+self.mean
        data[:,self.which_is_alpha] = 10**(data[:,self.which_is_alpha])  #Subtle variable alpha
        data[:,-1] = 10**(data[:,-1])  #Subtle variable damage
        return data
    def just_unormalize_alpha(self, _data_):
        data = copy.deepcopy(_data_)
        data = data*self.std[self.which_is_alpha] + self.mean[self.which_is_alpha]
        data = 10**(data)
        return data
    def just_unormalize_output(self, _data_):
        data = copy.deepcopy(_data_)
        data = data*self.std[-1] + self.mean[-1]
        data = 10**(data)
        return data

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--num_of_batch', type=int, default=50,
                        help='input batch size for training (default: 50)')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='number of epochs to train (default: 10000)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='how many mini-batches to wait before logging training status')
    parser.add_argument('--Visualization', action='store_true',\
            help="to show visualization, defualt false", default=False)
    parser.add_argument('--Do_savings', action='store_true',\
            help="if save every specific data", default=False)
    parser.add_argument('--Restart',  action='store_true',\
            help='if restart', default = False)
    parser.add_argument('--Debug', action='store_true',\
            help='debug mode show prints, default clean mode', default = False)
    parser.add_argument('--Time_step', type=int,\
            help="time step when run canadian model, default 24", default = 24)
    parser.add_argument('--Time_interval', type=float,\
            help="seperate '1 Hour' to 'this' steps, default 1", default = 1)
    parser.add_argument('--num_of_forces', type=int,\
            help="Number of forces, default 2", default = 2)
    parser.add_argument('--Number_of_woods', type=int,\
            help="Number of woods, default 100", default = 100)
    parser.add_argument('--Test_number_of_woods', type=int,\
            help="Test time number of woods, default 1000", default = 1000)
    parser.add_argument('--Years_to_run', type=float,\
            help="Years we run differential model", default = 5)
    parser.add_argument('--Test_years_to_run', type=float,\
            help="Years we run prediction NN model", default = 0.02)
    parser.add_argument('--workers', type=int,\
            help="Number of workers, default 10", default = 10)
    parser.add_argument('--Quick_data', action='store_true',\
            help='If Quick data from file, default False', default = False)
    parser.add_argument('--Save_model', action='store_true',\
            help='If Save model, default True', default = True)
    parser.add_argument('--Cuda_number', type=int,\
            help="Number of woods, default 0", default = 0)
    parser.add_argument('--RM', type=str,\
            help="Restart Model, default None, will use name current_best", default = None)
    parser.add_argument('--test_ratio', type=float,\
            help="Number of woods, default 0.3", default = 0.3)
    parser.add_argument('--wood_types', type=str,\
            help="Wood types, one of [All, Hemlock, SPF_Q1, SPF_Q2], default Hemlock", default = "Hemlock")
    parser.add_argument('--Alter_force_scaler', type=float,\
            help='If alter force when 1 year, default scaler 1.0, no alter', default = 1.0)
    args = parser.parse_args()
    assert args.Alter_force_scaler in [0.75, 1.0, 1.25], "Alter scaler recommended: [0.75, 1.0, 1.25]"
    if args.epochs<=0:   #If epoch <0, it's Test mode.
        args.Restart=True
        args.Quick_data=True
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    model_best_path = "../output/current_best_%s_%s_%s.pth"%(args.num_of_batch, args.wood_types, args.num_of_forces) if args.RM is None  else args.RM
    torch.manual_seed(44)
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(args.Cuda_number) 
    else:
        pass
    global wood_types
    if args.wood_types == "All":
        wood_types = collections.OrderedDict([
			("Hemlock", list(np.linspace(20.68, 31.03, args.num_of_forces))), 
			("SPF_Q1", list(np.linspace(30.34, 36.54, args.num_of_forces))), 
			("SPF_Q2", list(np.linspace(15.65, 18.13, args.num_of_forces))),
			])
        #wood_types = collections.OrderedDict([("Hemlock",[20.68, 31.03]), ("SPF_Q1",[30.34, 36.54]), ("SPF_Q2",[15.65*0.25, 18.13*0.25])])
    elif args.wood_types == "Hemlock":
       wood_types = collections.OrderedDict([
			("Hemlock", list(np.linspace(20.68, 31.03, args.num_of_forces))), 
			])
    elif args.wood_types == "SPF_Q1":
       wood_types = collections.OrderedDict([
			("SPF_Q1", list(np.linspace(30.34/2, 36.54*1.5, args.num_of_forces))), 
			])
    elif args.wood_types == "SPF_Q2":
       wood_types = collections.OrderedDict([
			("SPF_Q2", list(np.linspace(15.65, 18.13, args.num_of_forces))),
			])
    else:
        print("Unkown wood type")
        sys.exit()
    tmp_i = 0
    colors = ["red", "yellow", "orange", "green", "cyan", "blue"]*1000
    #colors = list(colors._colors_full_map.values())
    color_list = collections.OrderedDict()
    for wood_type in wood_types:
        for stress_level in wood_types[wood_type]:
            color_list[wood_type+str(stress_level)] =colors[tmp_i]
            tmp_i+=1

    #Get data:
    Quick_name = "Quick_pickle_%s_%s_%s_forces_%s_years_%s.pkl"%(args.Number_of_woods, args.wood_types, args.num_of_forces, args.Years_to_run, args.Alter_force_scaler)
    if args.Quick_data:
        objs = []
        print("Reading raw_data from pickle...%s"%Quick_name)
        with open("%s"%Quick_name, "rb") as pfile:
            while 1:
                try:
                    objs.append(pickle.load(pfile))
                except EOFError:
                    break
            raw_data, t_of_canadian_curves, canadian_curves = objs
    else:
        raw_data, t_of_canadian_curves, canadian_curves = differential_model.main_for_load_duration(args, wood_types)
        #Remove same raw_data to balance training:
        #Unique均衡移动至数据生成程序。
        #print("Unique balancing...")
        #_, idx = np.unique(raw_data, axis=0, return_index=True)
        #idx.sort()
        #raw_data_part1 = raw_data[idx]
        #print("Compensating...")
	#Compensating暂缓。
        #idx_those_removed = np.array(list(set(list(range(raw_data.shape[0])))-set(idx)))
        #np.random.shuffle(idx_those_removed)
        #raw_data_part2 = raw_data[idx_those_removed[:int(raw_data_part1.shape[0]*0.025)]]
        #raw_data_part2 = raw_data[idx_those_removed[:int(raw_data_part1.shape[0]*0)]]
        #raw_data = np.vstack((raw_data_part1, raw_data_part2))
        print("Dumping raw_data pickle...",)
        pfile = open("%s"%Quick_name, "wb")
        pickle.dump(raw_data, pfile)
        pfile.close()
        pfile = open("%s"%Quick_name, "ab")
        pickle.dump(t_of_canadian_curves, pfile)
        pickle.dump(canadian_curves, pfile)
        pfile.close()
    Preprocessor = data_preprocessing(raw_data)
    data = Preprocessor.clean(raw_data)
    Preprocessor.get_stastics(data)
    data= Preprocessor.normalize_all(data)
    inputs = torch.FloatTensor(data[:,:-1])
    targets = torch.FloatTensor(data[:,-1].reshape(-1, 1))
    train_dataset = Data.TensorDataset(inputs, targets)
    validate_dataset = Data.TensorDataset(inputs[::int(1/args.test_ratio)], targets[::int(1/args.test_ratio)])
    train_loader = Data.DataLoader( 
            dataset=train_dataset, 
            batch_size=int(len(train_dataset)/args.num_of_batch) if int(len(train_dataset)/args.num_of_batch)!=0 else len(train_dataset),
            shuffle=True,
            drop_last=True,
	    num_workers=args.workers,
            pin_memory=True
            )
    validate_loader = Data.DataLoader( 
            dataset=validate_dataset, 
            batch_size=int(len(validate_dataset)/args.num_of_batch) if int(len(validate_dataset)/args.num_of_batch)!=0 else len(validate_dataset),
            shuffle=True,
            drop_last=True,
	    num_workers=args.workers,
            pin_memory=True
            )
    if args.Visualization:
        for wood_type in wood_types.keys():
            for stress_level in wood_types[wood_type]:
                plt.plot(np.log10(t_of_canadian_curves[wood_type+str(stress_level)]), canadian_curves[wood_type+str(stress_level)])
        plt.title("%s, Altering force %s"%(args.wood_types, args.Alter_force_scaler))
        plt.show()
        pd.DataFrame(data).hist(bins=100)
        plt.show()
    if not args.Quick_data:
        print("Data Generation Done...")
        sys.exit()

    #Load Model:
    input_size = inputs.shape[1] 
    hidden_size = inputs.shape[1]*10*10
    hidden_depth = 5
    output_size = 1
    model = NeuralNet(input_size, hidden_size, hidden_depth, output_size, device).to(device)
    #model = RNNNet(input_size, hidden_size, hidden_depth, output_size, cell="RNN").to(device)
    #optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.Restart:
        model_restart_path = model_best_path if args.RM is None  else args.RM
        checkpoint = torch.load(model_restart_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_loss_history = checkpoint['train_loss_history']
        validate_loss_history = checkpoint['validate_loss_history']
        if args.Visualization:
            print("Restarting Model paramters loaded: %s"%model_restart_path)
            plt.plot(np.log10(validate_loss_history), linewidth=2, label="validate loss")
            plt.plot(np.log10(train_loss_history), linewidth=2, label="train loss")
            plt.title("Trainning/Validating loss over epoch")
            plt.legend()
            plt.show()
        else:
            pass
    else:
        epoch = 1
        train_loss_history = []
        validate_loss_history = []

    #Train and Validate:
    for epoch in range(epoch, args.epochs + 1):
        #print(model.bns[0].weight[-1].item(),model.fc1.weight[-1])
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        validate_loss, validate_outputs, validate_targets = validate(args, model, device, validate_loader)
        train_loss_history.append(train_loss)
        validate_loss_history.append(validate_loss)
        train_loss_history[0] = validate_loss_history[0]
        #plots:
        xaxis_train = range(len(train_loss_history))
        xaxis_validate = range(len(validate_loss_history))
        plt.clf()
        ax1 = plt.subplot(131)
        scope = 50
        ax1.scatter(xaxis_train[-scope:], 100*np.array(train_loss_history[-scope:]), color='k', s=0.85)
        ax1.scatter(xaxis_validate[-scope:], 100*np.array(validate_loss_history[-scope:]), color='blue', s=0.85)
        ax1.plot(xaxis_train[-scope:], 100*np.array(train_loss_history[-scope:]), color='k', label='trainloss')
        ax1.plot(xaxis_validate[-scope:], 100*np.array(validate_loss_history[-scope:]), color='blue', label="validateloss")
        ax1.legend()

        ax2 = plt.subplot(132)
        idx = np.argsort(validate_targets)
        skipper = int(len(validate_targets)/1000) if int(len(validate_targets)/1000)!=0 else 1
        ax2.scatter(range(len(validate_targets))[::skipper], validate_targets[idx][::skipper], color='k', label="validate_targets", s=0.15)
        ax2.scatter(range(len(validate_outputs))[::skipper], validate_outputs[idx][::skipper], color='blue', label='validate_outputs', s=0.15)

        ax3 = plt.subplot(133)
        ax3.scatter(range(len(validate_targets))[::skipper], Preprocessor.just_unormalize_output(validate_targets[idx])[::skipper], color='k', label="damage_targets", s=0.15)
        ax3.scatter(range(len(validate_outputs))[::skipper], Preprocessor.just_unormalize_output(validate_outputs[idx])[::skipper], color='blue', label='damage_outputs', s=0.15)
        ax2.legend()
        plt.draw()
        plt.pause(0.001)
        if epoch <= 1:
            continue
        if args.Save_model:
             torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss_history': train_loss_history,
                        'validate_loss_history': validate_loss_history,
                        }, "../output/epoch_%s_%s.pth"%(epoch, args.wood_types))
        if validate_loss_history[-1] < np.array(validate_loss_history[:-1]).min() and epoch>50:
            print("Saving current best model, epoch: %s"%epoch)
            os.system("cp ../output/epoch_%s_%s.pth %s"%(epoch, args.wood_types, model_best_path))
        else:
            print("Not best: %s>%s, Keep running..."%(validate_loss_history[-1], np.array(validate_loss_history[:-1]).min()))

    #Test:
    t_of_cum_lists, cum_lists, t_lists_for_alphas, real_alphas_lists = {}, {}, {}, {}
    for wood_type in wood_types:
        for test_force in wood_types[wood_type]:
            print("Testing on: %s of %spsi"%(wood_type, test_force))
            print("Testing time length: %s years, number of woods %s"%(args.Test_years_to_run, args.Test_number_of_woods))
            '''
            data = Preprocessor.unormalize_all(data)
            _,idx = np.unique(data[:, -2], axis=0, return_index=True)
            data = data[idx]
            data = data[np.where(data[:,Preprocessor.which_is_alpha]<0.01)]
            data = data[np.where(data[:,0]==test_force)][:args.Test_number_of_woods]
            test_input_data_easier = torch.FloatTensor(Preprocessor.normalize_all(data))[:,:3]
            '''
            test_data = differential_model.for_generalization_starting_point(args, wood_type=wood_type, force=test_force, start_alpha=1e-10)#1e-4)   #Leave pretreat for data_preprocessor
            test_data = np.hstack((test_data, np.tile(999,(test_data.shape[0],1))))
            test_input_data = torch.FloatTensor(Preprocessor.normalize_all(test_data))[:,:3]
            t_of_cum_lists[wood_type+str(test_force)], cum_lists[wood_type+str(test_force)], t_lists_for_alphas, real_alphas_lists[wood_type+str(test_force)] = test(args, model, device, test_input_data, Preprocessor, t_of_canadian_curves[wood_type+str(test_force)], canadian_curves[wood_type+str(test_force)], wood_type, test_force, color_list)
    plt.figure()
    RRs = np.array([])
    savings = pd.DataFrame([])
    if args.wood_types=="SPF_Q1":
        wood_types['SPF_Q1'] = wood_types['SPF_Q1'][20:-20]
    embed()
    for wood_type in wood_types:
        for stress_level in wood_types[wood_type]:
            #canadian_curves[wood_type+str(stress_level)] = np.insert(canadian_curves[wood_type+str(stress_level)],0,0)
            #cum_lists[wood_type+str(stress_level)] = np.insert(cum_lists[wood_type+str(stress_level)],0,0)
            #t_of_canadian_curves[wood_type+str(stress_level)] = np.insert(t_of_canadian_curves[wood_type+str(stress_level)],0,1)
            #t_of_cum_lists[wood_type+str(stress_level)] = np.insert(t_of_cum_lists[wood_type+str(stress_level)],0,1)
            tmp_len = min(canadian_curves[wood_type+str(stress_level)].shape[0], cum_lists[wood_type+str(stress_level)].shape[0])
            canadian_curves[wood_type+str(stress_level)] = canadian_curves[wood_type+str(stress_level)][:tmp_len]
            cum_lists[wood_type+str(stress_level)] = cum_lists[wood_type+str(stress_level)][:tmp_len]
            t_of_canadian_curves[wood_type+str(stress_level)] = t_of_canadian_curves[wood_type+str(stress_level)][:tmp_len]
            t_of_cum_lists[wood_type+str(stress_level)] = t_of_cum_lists[wood_type+str(stress_level)][:tmp_len]
            RR = np.round(np.corrcoef(canadian_curves[wood_type+str(stress_level)], cum_lists[wood_type+str(stress_level)])[0,1]**2*100, 2)
            plt.plot(np.log10(t_of_canadian_curves[wood_type+str(stress_level)]), canadian_curves[wood_type+str(stress_level)], '--', label="Canadian Model of %s %s"%(wood_type, np.round(stress_level, 2)), color=color_list[wood_type+str(stress_level)])
            plt.plot(np.log10(t_of_cum_lists[wood_type+str(stress_level)]), cum_lists[wood_type+str(stress_level)], label="Network Result of %s %s, D:%s%%"%(wood_type, np.round(stress_level, 2), RR), color=color_list[wood_type+str(stress_level)])
            RRs = np.append(RRs, RR)
            savings["canadian_t_of_"+wood_type+str(stress_level)] = pd.Series(np.log10(t_of_canadian_curves[wood_type+str(stress_level)]))
            savings["canadian_curve_of_"+wood_type+str(stress_level)] = pd.Series(canadian_curves[wood_type+str(stress_level)])
            savings["Network_t_of_"+wood_type+str(stress_level)] = pd.Series(np.log10(t_of_cum_lists[wood_type+str(stress_level)]))
            savings["Network_curve_of_"+wood_type+str(stress_level)] = pd.Series(cum_lists[wood_type+str(stress_level)])
    savings.to_csv("Final_Performance", index=None)
    plt.legend(loc="upper left", fontsize=8)
    plt.title("Network prediction over all types of wood and all load with mean D of %s%%"%np.round(RRs.mean(), 2))
    plt.show()

if __name__ == '__main__':
    np.random.seed(55)
    main()
