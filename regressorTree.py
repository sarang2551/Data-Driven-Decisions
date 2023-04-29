from utility import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import  train_test_split, GridSearchCV

def get_best_model_TreeRegressor(X,y,score_function,cv=3,verbose=3):
    xtrain, xtest, ytrain, ytest = train_test_split(X,y,test_size=0.25,random_state=100)
    best_model = None
    parameters={"splitter":["best","random"],
            "max_depth" : [1,3,5,7,9,11,12],
           "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
           "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5],
           "max_features":["auto","log2","sqrt",None],
           "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] }
    grid = GridSearchCV(DecisionTreeRegressor(random_state=100),param_grid=parameters,scoring=score_function(),cv=3,verbose=3)
    grid.fit(X,y)
    best_params = grid.best_params_
    print(best_params)
    best_model = DecisionTreeRegressor(random_state=100,
                                       max_depth=best_params["max_depth"],
                                       max_features=best_params["max_features"],
                                       max_leaf_nodes=best_params["max_leaf_nodes"],
                                       min_samples_leaf=best_params["min_samples_leaf"],
                                       min_weight_fraction_leaf=best_params["min_weight_fraction_leaf"],
                                       splitter=best_params["splitter"]
                                       )
    best_model.fit(X,y)
    return best_model

def run_step_mss(data,doPrint=False):
    is_x,is_c = data['in_sample_x'],data['in_sample_y']
    ofs_x, ofs_c = data['out_sample_x'], data['out_sample_y']
    best_model = get_best_model_TreeRegressor(is_x,is_c,MSE)
    actual_costs = []
    true_costs = []
    differences = []
    for idx in range(len(ofs_x)):
        x_vec = np.array(ofs_x[idx]).reshape(1,5)
        cost_vec = np.array(ofs_c[idx]).reshape(1,40)
        #beta = best_model.coef_
        y_pred = best_model.predict(x_vec).reshape(40,1)
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

def run_step_acpp(data,doPrint=False):
    is_x,is_c = data['in_sample_x'],data['in_sample_y']
    ofs_x, ofs_c = data['out_sample_x'], data['out_sample_y']
    best_model = get_best_model_TreeRegressor(is_x,is_c,ACPP,cv=1)
    costs = []
    optimal_cost = []
    differences = []
    for idx in range(len(ofs_x)):
        actual_cost = np.array(ofs_c[idx]).reshape(1,40)
        x_vec = np.array(ofs_x[idx]).reshape(1,5)
        y_pred = best_model.predict(x_vec).reshape(40,1)
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
def run_experiment(n,d=1,e=0.5):
    in_sample_x,in_sample_y = make_data(n,degree=d,e=e)
    out_sample_x,out_sample_y = make_data(1000,d,e)
    data = {'in_sample_x':in_sample_x,'in_sample_y':in_sample_y,
            'out_sample_x':out_sample_x,'out_sample_y':out_sample_y}  
    run_step_mss(data,True)
    run_step_acpp(data,True)

run_experiment(100,1,0.5)