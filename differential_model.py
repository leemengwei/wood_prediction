import tqdm
import argparse
from IPython import embed
import warnings 
warnings.filterwarnings("ignore")
import os,sys,time
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib
import itertools
def parser_args():
    parser = argparse.ArgumentParser(description = "LogT-Damage Net...")
    parser.add_argument('--Visualization', '-V', action='store_true',\
            help="to show visualization, defualt false", default=False)
    parser.add_argument('--Number_of_woods', '-N', type=int,\
            help="Number of woods in one type", default=100)
    parser.add_argument('--Debug', '-D', action='store_true',\
            help='debug mode show prints, default clean mode', default = False)
    parser.add_argument('--Time_interval', '-I', type=float,\
            help="seperate '1 Hour' to 'this' steps, default 10", default = 10)
    parser.add_argument('--Test_number_of_woods', type=int,\
            help="Test time number of woods, default 1000", default = 1000)
    parser.add_argument('--Years_to_run', type=float,\
            help="Years we run differential model (in train/val) and to predict using NN model", default = 5)
    args = parser.parse_args()
    return args

def get_force_pure_short(t, Ramp_rate):
    force = Ramp_rate*t*1e-5 
    return force

def get_force_pure_long(stress_level):
    return stress_level

def get_force_alter(t, stress_level):
    if t>100:
        return stress_level*100
    else:
        return stress_level

def get_force_short_and_long(t, stress_level, Ramp_rate):
    force = Ramp_rate*t
    if force < stress_level:
        return force
    else:
        return stress_level

def get_a_normal_value_by_distribution(Mean, Cov):
    value = np.random.normal(Mean, Cov)
    return value

def get_a_lognormal_value_by_distribution(Mean, Cov):
    lognormal_mean = Mean
    lognormal_std = Cov*Mean
    #Conversion referring to https://stats.stackexchange.com/questions/95498/how-to-calculate-log-normal-parameters-using-the-mean-and-std-of-the-given-distr
    normal_std = np.sqrt(np.log(1 + (lognormal_std/lognormal_mean)**2))
    normal_mean = np.log(lognormal_mean) - normal_std**2 / 2
    value = np.random.lognormal(normal_mean, normal_std)
    return value

def get_constants(wood_type):
    #print("Calling for new params...")
    if wood_type == "Hemlock":
        b = get_a_lognormal_value_by_distribution(37.161, 0.281)
        c = get_a_lognormal_value_by_distribution(1.623*10e-4, 0.574)
        n = get_a_lognormal_value_by_distribution(1.290, 0.075)
        sigma0 = get_a_lognormal_value_by_distribution(0.533 ,0.298)
        tao_s = get_a_lognormal_value_by_distribution(47.83, 0.41)
        Ramp_rate = 2678.63
    elif wood_type == "SPF_Q1":
        b = get_a_lognormal_value_by_distribution(77.392, 0.174)
        c = get_a_lognormal_value_by_distribution(2.810*10e-4, 0.824)
        n = get_a_lognormal_value_by_distribution(1.162, 0.231)
        sigma0 = get_a_lognormal_value_by_distribution(0.420, 0.038)
        tao_s = get_a_lognormal_value_by_distribution(48.90, 0.20)
        Ramp_rate = 2675.18
    elif wood_type == "SPF_Q2":
        b = get_a_lognormal_value_by_distribution(158.656, 0.009)
        c = get_a_lognormal_value_by_distribution(4.317*10e-4, 0.745)
        n = get_a_lognormal_value_by_distribution(1.285, 0.170)
        sigma0 = get_a_lognormal_value_by_distribution(0.365, 0.562)
        tao_s = get_a_lognormal_value_by_distribution(25.77, 0.28)
        Ramp_rate = 1647.86
    else:
        print("Unknown type... exiting...")
        sys.exit(0)
    return b, c, n, sigma0, tao_s, Ramp_rate

def step_model_core(alpha, tao_t, delta_t, params):
    if alpha >= 1:
        return alpha
    b, c, n, sigma0, tao_s, Ramp_rate = params
    if tao_t-sigma0*tao_s<0:    #currently not enough
        return alpha
    if sigma0>1:
        b_plus_1 = int(np.round(b+1)-np.round(b+1)%2)
    else:
        b_plus_1 = b+1
    Ki = np.exp(c*(tao_t-sigma0*tao_s)**n*delta_t)
    try:
        a = (Ramp_rate*(b_plus_1)) / ((tao_s-tao_s*sigma0)**(b_plus_1))  #formula given in paper 
        Li = (a/c)*(tao_t-sigma0*tao_s)**(b-n)*(Ki-1)
    except:
        return 1     #division by 0, rare ocassion, just return 1
    if np.isfinite(Li)&np.isfinite(Ki)&np.isfinite(alpha)&(~np.iscomplex(alpha)):
        alpha = alpha*Ki + Li
    else:
        return 1     #inf, nan or complex ocassion, just return 1
    alpha = 1 if alpha>1 else alpha
    return alpha

def damage_model_discontinous(t_all, stress_level, alpha_start, params, args):
    alpha_list, t_list, force_list = [], [], []
    alpha = alpha_start
    alpha_stop = 1.0
    count = -1
    t_previous = 0
    b, c, n, sigma0, tao_s, Ramp_rate = params
    for t in t_all:
        delta_t = t - t_previous
        t_previous = t
        count+=1
        #tao_t = get_force_short_and_long(t, stress_level, Ramp_rate)
        tao_t = get_force_pure_long(stress_level)
        if args.Alter_force_scaler!=1 and t>24*365*1:  #Alter force when 1 year to 1.25 times.
            tao_t = tao_t*args.Alter_force_scaler
            pass
        #tao_t = get_force_alter(t, stress_level)
        #MODEL CORE:
        alpha = step_model_core(alpha, tao_t, delta_t, params)
        if count%(1e5*3)==0 and count!=0:
            print("Computing report, Alpha:%s, Time:%s, Force:%s"%(alpha, t, tao_t))
        alpha_list.append(alpha)
        t_list.append(t)
        force_list.append(tao_t)
        if alpha >= 1:
            break
    #alpha_list.insert(0, alpha_start)   #Since t_all doesn't include time 0, we mannually include data for time 0 here.
    #t_list.insert(0, 1e-5)    #t need to be >0 due to later log act.
    #force_list.insert(0, force_list[0])
    return alpha_list, t_list, force_list

def get_interp_values_of_x_y(num_of_woods, nn_input_dim, t_list, y_list, f_list,  aux_list):
    ##interp_step = 1
    sample_step = 1
    '''
    inputs = np.empty(shape=(0, nn_input_dim+1))
    for case_index, i in enumerate(range(len(t_list))):
        print("Interpolating over:%s"%case_index)
        t_sample = t_list[i][::sample_step]
        if len(t_sample) == 0:
            continue
        y_sample = np.array(y_list[i][::sample_step])
        diff_sample = np.diff(y_sample)
        f_sample = np.array(f_list[i][::sample_step])
        aux_nointerp = np.tile(aux_list[case_index], y_sample.__len__())
        #INPUTS Other variables:
        try:
            inputs = np.vstack((inputs, np.vstack((f_sample[:-1], y_sample[:-1], aux_nointerp[:-1], diff_sample)).T))
        except:
            pass
    '''
    diff_list = []
    for i in range(len(t_list)):
        diff_list.append(np.diff(y_list[i])[::sample_step])
        y_list[i] = y_list[i][:-1][::sample_step]
        f_list[i] = f_list[i][:-1][::sample_step]
        aux_list[i] = np.tile(aux_list[i], len(y_list[i]))
    Ys = list(itertools.chain(*y_list))
    Fs = list(itertools.chain(*f_list))
    Auxs = list(itertools.chain(*aux_list))
    Diffs = list(itertools.chain(*diff_list))  #注：无法移动至preprocess，因为数据分段，而前后又无明确单调关系
    inputs = np.stack((Fs,Ys,Auxs,Diffs)).T
    print("Unique balancing...")
    _, idx = np.unique(inputs, axis=0, return_index=True)
    idx.sort()
    inputs = inputs[idx]
    return inputs

def main_for_load_duration(args, wood_types):
    num_of_woods = args.Number_of_woods
    resolution_subtle = args.Time_interval
    resolution_coarse = 1
    subtle_end = 100
    end_t = 24*365*args.Years_to_run
    t_subtle = np.linspace(0, subtle_end, subtle_end*resolution_subtle+1)
    t_coarse = np.linspace(subtle_end+1/resolution_coarse, end_t+subtle_end, end_t*resolution_coarse)
    #t_all = np.append(t_subtle,t_coarse)
    t_all = np.arange(0, end_t, args.Time_step)
    alpha_start = 1e-10
    alpha_stop = 1.0
    nn_input_dim = 3
    datas = {}
    cum_curves = {}
    t_of_cum_curves = {}
    data = np.empty(shape=(0,nn_input_dim+1))
    for wood_type in wood_types.keys():
        wood_params_of_a_type = []
        broken_rates_of_a_type = []
        y_list_of_a_type = []
        t_list_of_a_type = []
        f_list_of_a_type = []
        for stress_level in wood_types[wood_type]:
            print("Sampling on:", wood_type, stress_level)
            broken_rates = []
            broken_ts = []
            wood_params = []
            y_list = []
            t_list = []
            f_list = []
            for case_index, case in enumerate(range(num_of_woods)):
                params = get_constants(wood_type)
                wood_params.append(params[-2])
                print("*"*5, "WoodType:",wood_type, "on Mpa:",stress_level,"case:",case,"b,c,n,sigma0,tao_s,Ramp_rate:",params)
                y, t, f = damage_model_discontinous(t_all, stress_level, alpha_start, params, args)
                y_list.append(y)
                t_list.append(t)
                f_list.append(f)
                y_list_of_a_type.append(y)
                t_list_of_a_type.append(t)
                f_list_of_a_type.append(f)
                broken_rate = y[-1]/t[-1]
                broken_rates.append(broken_rate)
                broken_t = t[-1]
                broken_ts.append(broken_t)
            broken_rates = np.array(broken_rates)
            broken_ts = np.array(broken_ts)
            count, division = np.histogram(broken_ts, t_all[1:])
            cum_curves[wood_type+str(stress_level)] = np.cumsum(count[:-1])/num_of_woods
            t_of_cum_curves[wood_type+str(stress_level)] = division[1:-1]
            broken_rates_of_a_type.extend(broken_rates)
            sorted_idx_rates = np.argsort(broken_rates)  #slow to fast
            wood_mean = np.array(wood_params).mean()
            wood_cov = np.array(wood_params).std()/wood_mean
            random_quality = get_a_lognormal_value_by_distribution(np.tile(wood_mean, num_of_woods), np.tile(wood_cov, num_of_woods))
            random_quality.sort()
            corresponding_quality = list(random_quality[np.argsort(sorted_idx_rates[::-1])])
            #Get interpted data as NN inputs:
            print("Interpolating and generating data for NN.")
            data = np.vstack((data, get_interp_values_of_x_y(num_of_woods, nn_input_dim, t_list, y_list, f_list, corresponding_quality)))

        if args.Visualization:
            for stress_level in wood_types[wood_type]:
                #Collections of individual damage:
                #for i in range(len(t_list)):
                #    plt.plot(t_list[i], np.log10(y_list[i]))
                #plt.title("Transformed_{num_w}_wood_damage_collection_of_{wt}_under_{sl}".format(num_w=num_of_woods, wt=wood_type, sl=stress_level))
                #plt.savefig("../data/Wood_damage_collection_of_{num_w}_{wt}_{sl}_no_mutation.png".format(num_w=num_of_woods, wt=wood_type, sl=stress_level))
                #plt.show()
                #Culmulation curves:
                plt.plot(np.log10(t_of_cum_curves[wood_type+str(stress_level)]), cum_curves[wood_type+str(stress_level)])
            plt.title("%s, %s~%s Mpa"%(wood_type, wood_types[wood_type][0], wood_types[wood_type][-1]))
            plt.show()
        plt.clf()
    print("Generate data shape:",data.shape)
    #embed()
    return data, t_of_cum_curves, cum_curves

def read_constants_and_quality_order(wood_type):
    #quality order is given from bad to good.
    return
'''
def main_for_sample():
    wood_type_list = ["Hemlock", "SPF_Q1", "SPF_Q2"]
    wood_type = wood_type_list[0]
    #for wood_type in wood_type_list:
    case_num=100
    for case in range(case_num):
        print("Case:",case)
        params, quality = read_constants_and_quality_order(wood_type)
        alpha_previous = 0.5
        force_list = np.arange(5,50,1)
        for force in force_list:
            alpha_previous_list = np.round(np.arange(0,1,0.1),6)
            for alpha_previous in alpha_previous_list:
                alpha_next = step_model_core(alpha_previous, force, 0.0001, params)
                alpha_delta = np.round(alpha_next - alpha_previous,6)
                print(wood_type, "previous:", alpha_previous, "delta:", alpha_delta, "force:", force)
    return
'''

def for_generalization_starting_point(args, wood_type, force, start_alpha=1e-10):
    print("Getting starting points with %s of %s..."%(wood_type, args.Test_number_of_woods))
    wood_params = []
    for i in range(args.Test_number_of_woods):
        params = get_constants(wood_type)
        wood_param = params[-2]
        wood_params.append(wood_param)
    wood_params = np.array(wood_params)
    mean = wood_params.mean()
    cov = wood_params.std()/mean
    start_alphas = np.tile(start_alpha, args.Test_number_of_woods)
    forces = np.tile(force, args.Test_number_of_woods)
    aux_quality = get_a_lognormal_value_by_distribution(np.tile(mean, args.Test_number_of_woods), np.tile(cov, args.Test_number_of_woods))
    datas = np.vstack((forces, start_alphas, aux_quality)).T
    return datas

if __name__ == "__main__":
    np.random.seed(5)
    if os.path.isfile("./log.txt"):
        print("Removing old log...")
        os.remove("./log.txt")

    args = parser_args()
    #wood_types = {"Hemlock":[20.68, 31.03], "SPF_Q1":[30.34, 36.54], "SPF_Q2":[15.65, 18.13]}
    #wood_types = {"Hemlock":[999], "SPF_Q1":[999], "SPF_Q2":[999]}
    wood_types = {"Hemlock":[20.68, 31.03]}

    datas, t_of_cum_curve, cum_curve = main_for_load_duration(args, wood_types)
    #main_for_sample()
    datas = for_generalization_starting_point(args, 31.04, 1e-10)

