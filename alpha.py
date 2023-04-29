
import numpy as np
from utility import *
import random
from sklearn.model_selection import  train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, Ridge, ElasticNet

import sys
from math import *


def downwards_optimisation(X,y,ML_MODEL):
    # take in x_train to train n models 
    # use x_test, y_test to get beta_namptha values
    # use those beta namptha values to generate predictions
    # use predictions to make paths and then calculate AC * PP 
    # use AC * PP to get namptha value that generates the lowest cost 
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=100)
    namptha_vals = np.arange(0.005,0.5,0.005)
    
    parameters = {'alpha':namptha_vals}
    # best_model = None
    # min_diff = sys.maxsize
    # costs_arr = []
    # for alpha in namptha_vals:
    #     model = ML_MODEL(alpha=alpha)
    #     model.fit(x_train,y_train)
    #     y_pred = model.predict(x_test) # 2D vector with shape (0.25*n,40)
    #     actual_cost = 0 # for a particular alpha val what is the predicted cost 
    #     for idx in range(len(y_pred)): # for every 40 dimensional vector
    #         predictedVec = np.array(y_pred[idx]).reshape(40,1)
    #         g = make_graph(predictedVec)
    #         shortestPredictedPath = np.array(makeBinaryVector(get_shortest_path(g))).reshape(40,1)
    #         # calculate actual cost * shortest predicted path
    #         actual_cost += y_test[idx]@shortestPredictedPath
    #     if actual_cost < min_cost:
    #         min_cost = actual_cost
    #         best_model = model
    #     costs_arr.append(actual_cost)
    grid = GridSearchCV(ML_MODEL(),param_grid=parameters,scoring=ACPP(),cv=3,verbose=3)
    grid.fit(X,y)
    best_namptha = grid.best_params_['alpha']
    best_model = ML_MODEL(alpha=best_namptha)
    best_model.fit(X,y)
    return best_model
    # plt.plot(namptha_vals,costs_arr)
    # plt.axvline(x=best_model.alpha,c="orange",label="Minimum point line")
    # plt.xlabel("Namptha values")
    # plt.ylabel("Costs")
    # plt.title("Metric: AC* PP")
    # plt.show()
    # print(f"Best model namptha: {best_model.alpha}")
    return best_model


def get_best_model_alpha(c,x,ML_MODEL):
    xtrain, xtest, ytrain, ytest = train_test_split(x,c,test_size=0.25,random_state=100)
    namptha_vals = np.arange(0.01,1,0.01)
    parameters = {'alpha':namptha_vals}
    # error_vals = []
    # err_min = sys.maxsize
    # best_model = None
    # for alpha in namptha_vals:
    #     model = ML_MODEL(alpha=alpha)
    #     model.fit(xtrain,ytrain)
    #     ypred = model.predict(xtest) #n/4, 5
    #     err = mean_squared_error(y_true=ytest,y_pred=ypred)	# choosing metric
    #     if err < err_min:
    #         err_min = err
    #         best_model = model
    #     error_vals.append(err)
    #fitted_model = ML_MODEL().fit(x,c)
    grid = GridSearchCV(ML_MODEL(),param_grid=parameters,scoring=MSE(),cv=3,verbose=3)
    grid.fit(x,c)
    best_namptha = grid.best_params_['alpha']
    best_model = ML_MODEL(alpha=best_namptha)
    best_model.fit(x,c)
    return best_model

def run_step1(data,ml_model,doPrint=False):
    is_x,is_c = data['in_sample_x'],data['in_sample_y']
    ofs_x, ofs_c = data['out_sample_x'], data['out_sample_y']
    # optimize using Mean squared error to get best model
    mss_best_model = get_best_model_alpha(c=is_c,x=is_x,ML_MODEL=ml_model)
    actual_costs = []
    true_costs = []
    differences = []
    for idx in range(len(ofs_x)):
        x_vec = np.array(ofs_x[idx]).reshape(5,1)
        cost_vec = np.array(ofs_c[idx]).reshape(1,40)
        beta = mss_best_model.coef_
        y_pred = np.dot(beta,x_vec)
        path = get_shortest_path(make_graph(y_pred))
        binaryVec = np.array(makeBinaryVector(path)).reshape(40,1)
        actual_cost = cost_vec@binaryVec # AC * PP (Stage 1)

        optimal_path = get_shortest_path(make_graph(cost_vec.reshape(40,1)))
        optimal_binary_path = makeBinaryVector(optimal_path)
        true_cost = cost_vec@np.array(optimal_binary_path).reshape(40,1) # actual optimal cost (actual_cost*optimalPath)
        #cost_vec = list(map(float,cost_vec.reshape(40,1)))
        # if actual_cost < true_cost:
        #     return Exception(f"Actual cost lower than optimal cost using MSS: AC {actual_cost}, OC: {true_cost}")
        actual_costs.append(float(actual_cost))
        true_costs.append(float(true_cost))
        differences.append(abs(actual_cost-true_cost))

    actual_costs, differences = np.array(actual_costs), np.array(differences)
    if doPrint:
        print("Metric: Mean Squared Error")
        print(f"Mean actual_pred_cost stage 1: {round(np.mean(actual_costs),2)}")
        print(f"Mean standard deviation 1: {round(np.std(actual_costs),2)}")
        print(f"Median stage 1: {round(np.median(actual_costs),2)}")
        print(f"Mean difference stage 1: {round(np.mean(differences),2)}")   
    return actual_costs

def run_step2(data,ML_MODEL,doPrint=False):
    X,y = data['in_sample_x'],data['in_sample_y']
    ofs_x, ofs_c = data['out_sample_x'], data['out_sample_y']
    # optimize model using ACPP cost as the metric
    best_model = downwards_optimisation(X,y,ML_MODEL)
    beta = best_model.coef_
    costs = []
    optimal_cost = []
    differences = []
    for idx in range(len(ofs_x)):
        actual_cost = np.array(ofs_c[idx]).reshape(1,40)
        x_vec = np.array(ofs_x[idx]).reshape(5,1)
        y_pred = np.dot(beta,x_vec)
        predicted_path = get_shortest_path(make_graph(y_pred))
        binary_path = np.array(makeBinaryVector(predicted_path)).reshape(40,1)

        optimal_path = get_shortest_path(make_graph(actual_cost.reshape(40,1)))
        optimal_binary_path = np.array(makeBinaryVector(optimal_path)).reshape(40,1)

        ac_pp = float(actual_cost@binary_path)
        ac_op = float(actual_cost@optimal_binary_path)
        # if ac_pp < ac_op:
        #     print(f"Error in step 2!")
        #     return Exception(f"Actual Cost lower than optimised cost using ACPP: AC {ac_pp} , OC {ac_op}")
        costs.append(ac_pp)
        optimal_cost.append(ac_op)
        differences.append(ac_pp-ac_op)
    if doPrint:
        print("Metric: AC*PP - AC*OP")
        print(f"Mean actual_pred_cost stage 2: {round(np.mean(costs),2)}")
        print(f"Mean standard deviation stage 2: {round(np.std(costs),2)}")
        print(f"Median stage 2: {round(np.median(costs),2)}")
        print(f"Mean difference stage 2: {round(np.mean(differences),2)}") 
    return costs

def run_experiment(n,d,e,model,iterations = 1):
    avg_mss_costs = []
    avg_acpp_costs = []
    in_sample_x,in_sample_y = make_data(n,degree=d,e=e)
    out_sample_x,out_sample_y = make_data(1000,d,e)
    data = {'in_sample_x':in_sample_x,'in_sample_y':in_sample_y,
            'out_sample_x':out_sample_x,'out_sample_y':out_sample_y}   
    MSS_COSTS = run_step1(data,model,False)
    ACPP_COSTS = run_step2(data,model,False)
    avg_mss_costs.append(np.mean(MSS_COSTS))
    avg_acpp_costs.append(np.mean(ACPP_COSTS))
    
    print(f"Mean MSS costs: {np.mean(np.array(avg_mss_costs))}")
    print(f"Mean ACPP costs: {np.mean(np.array(avg_acpp_costs))}")

    #return np.mean(np.array(avg_mss_costs)), np.mean(np.array(avg_acpp_costs))
    
    makeHists(mss=MSS_COSTS,acpp=ACPP_COSTS,title=f"Parameters n:{n} d:{d} e:{e}, model: Lasso")
run_experiment(1000,1,0.5,model=Lasso)
# mss_costs = []
# acpp_costs = []
# n_vals = list(range(10,1001))
# model = Lasso
# for n in n_vals:
#     mss,acpp = run_experiment(n,1,0.5,model=model)
#     mss_costs.append(mss)
#     acpp_costs.append(acpp)
# plt.plot(n_vals,mss_costs,label="MSS_Costs")
# plt.plot(n_vals,acpp_costs,label="ACPP_Costs",c='orange')
# plt.legend(["MSS_Costs","ACPP_Costs"])
# plt.ylabel("Cost")
# plt.xlabel("n")
# plt.title(f" Model used: Lasso")
# plt.show()