# Data-Driven-Decisions
A quantitative analysis on the performance of decision aware models.
## Background
The ubiquitous presence of ambiguity and unpredictability pervades every real world decision and thus the Decision Maker (DM) is confronted with an element of unpredictability, whether it be within the objective function that he seeks to maximize, or in some of the restrictions that he must adhere to. With the abundance of data now, many approaches are being formulated that seek to develop models that aid in achieving the most optimal solutions. One way to for model optimisation that diverge from main stream approaches is through contextually optimising models. This is where downstream data driven decisions come into play and where models become more "decision aware". 

This problem is futher illustrated in the paper "Data-Driven Conditional Robust Optimization" (refered below). In a standard cost minimization problem, where X ⊂ ℝ⁵ and cost(x,ξ). X captures the feasible set of actions and the cost, cost(x,ξ), that depends on both the action and a random permutation vector, ξ ( ξ ∈ $\mathbb{R}^m$). 

$\min_{x(\cdot)} \mathbb{E}[c(x(\psi), \xi)]$&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(1)
            
            
The standard approach to the cost minimization problem would be to predict then optimize. In general, machine learning tools are designed to minimize errors in prediction. How the predictions will be utilized in the optimization problem downstream is usually out of the equation. 

In contrast, this project explores the "Smart, Predict, then Optimize" framework that is introduced in the Management Science paper by  Adam N.Elmachtoub and Paul Grigas. This model directly utilizes the base framework of the optimization problem such as the problem's objective and constraints and produces a better prediction model.

In this project report, the perfomance of this approach in being quantitatively explored.  

[Experiment 1](#experiment1)
[Data generation](#data_generation)

This particular implementation of synthetic data generation is taken from the Elmachtoub-Grigas paper.
 
```python
def make_data(n, degree = 1,e = 0.5):
    b = bernoulli(0.5)
    d = 40 # total number of edges in a 5 by 5 matrix
    p = 5 # number of features
    # Construct a x feature vector
    I_p = np.identity(5)
    x = np.random.multivariate_normal(mean=np.array([0,0,0,0,0]),cov=I_p,size=(n))
    for i in range(n):
        for j in range(p):
            x[i][j] = abs(x[i][j])
    B = np.zeros(shape=(d,p))
    c = np.zeros(shape=(n,d))
    for i in range(d):
        for j in range(p):
            B[i][j] = b.rvs()
    
    # Construct a cost vector with dimension d
    for i in range(n):
        x_i = x[i].reshape(5,1)
        for j in range(d):
            noise = np.random.uniform(low=1-e,high=1+e)
            frac = 1/np.sqrt(p)
            vec_mult = (B @ x_i)[j]
            c[i][j] = (((frac*vec_mult + 3)**degree)+1)*noise
    return x,c
```

The vector x that represents features is in $\mathbb{R}^p$ while the cost vector, c, is in $\mathbb{R}^d$. In the function above, p and d are hardcoded to be 5 and 40 respectively. The dimension for the cost vector, d, correponds to the number of edges in a 5 by 5 matrix, other dimensions can be used for a n by m matrix using the following formula: Edges = n * (m-1) + m * (n-1). First, a random matrix $\mathbb{B}^* \subseteq \mathbb{R}^{d \times p}$ is generated which encodes the parameters of the true model, where each entry, b, in $\mathbb{B}^*$ is $b \sim \text{Bernoulli}(0.5)$. After the $\mathbb{B}^*$ matrix the training data is generated with n entries in total. Each entry has a feature vector x and a cost vector.

This function is used to generate both in-sample training data as well as out-of-sample test data.  

[Mean Squared Error Metric](#mss_metric)

The first section of experiment one focuses on the standard approach to machine learning models which is to "Predict then optimize". Data is generated and then split for training and testing purposes. A Lasso model is trained on the synthetic data and a namptha value is determined using Mean Square Error (MSE) as the metric. Basically finding the optimum namptha value ,$\lambda^*$, such that the Lasso model $\hat{\beta}^{lasso} = \underset{\beta}{argmin}\left\lbrace\frac{1}{2n}||\mathbf{y}-\mathbf{X}\beta||_2^2 + \lambda ||\beta||_1 \right\rbrace$ results in the lowest MSE. 

Code for obtaining the best model with MSE metric: 
```python
def get_best_model_alpha():
    x,c = make_data(1000)
    xtrain, xtest, ytrain, ytest = train_test_split(x,c,test_size=0.25,random_state=100)
    namptha_vals = np.arange(0.001,1,0.001)
    error_vals = []
    err_min = sys.maxsize
    best_model = None
    for alpha in namptha_vals:
        model = Lasso(alpha=alpha)
        model.fit(X=xtrain,y=ytrain)
        ypred = model.predict(X=xtest) #n/4, 5
        err = mean_squared_error(y_true=ytest,y_pred=ypred)
        if err < err_min:
            err_min = err
            best_model = model
        error_vals.append(err)
    return best_model, error_vals, namptha_vals
```

![Graph to illustrate the minimum error possible for different Lasso models utilising different namptha values](minMSS_graph.png)

The minimum point on the graph above represents the optimal model with $\lambda^*$ as its parameter. 

[Downstream decision](#downstream_decision_exp1)

For this experiment we will be looking at the shortest path as a potential downstream decision, more explictly the cost associated to the shortest path, given a $5\times 5$ matrix with 40 edges. Each $i^\text{th}$ entry, c[i], in the cost vector will represent the cost associated to traversing that edge. 

This is where there is a change in the metric for assessing models. Initially the MSE was used to obtain $\lambda^*$, now the metric would be the cost associated with traversing the shortest path in a $5\times 5$ matrix.

```python
def downwards_optimisation(X,y,ML_MODEL):
    # take in x_train to train n models 
    # use x_test, y_test to get beta_namptha values
    # use those beta namptha values to generate predictions
    # use predictions to make paths and then calculate cost associated with the shortest path
    # use the cost to get namptha value that generates the lowest cost 
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=100)
    namptha_vals = np.arange(0.005,0.5,0.005)
    best_model = None
    min_cost = sys.maxsize
    costs_arr = []
    for alpha in namptha_vals:
        model = ML_MODEL(alpha=alpha)
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test) # 2D vector with shape (0.25*n,40)
        actual_cost = 0 # for a particular alpha val what is the predicted cost 
        for idx in range(len(y_pred)): # for every 40 dimensional vector
            predictedVec = np.array(y_pred[idx]).reshape(40,1)
            g = make_graph(predictedVec)
            shortestPredictedPath = np.array(makeBinaryVector(get_shortest_path(g))).reshape(40,1)
            # calculate actual cost * shortest predicted path
            actual_cost += y_test[idx]@shortestPredictedPath
        if actual_cost < min_cost:
            min_cost = actual_cost
            best_model = model
        costs_arr.append(actual_cost)
    return best_model
```

The $\textit{make\_graph}$ function converts the 40 dimensional matrix into a graph where the edges c[0], c[1], c[2] ... c[39] are edges connecting the 25 vertices. The $\textit{get\_shortest\_path}$ function utilises Dijkstra's algorithm to find the shortest path in the graph which the $\textit{makeBinaryVector}$ function uses to create a vector with 0s and 1s representing which edge is traversed in the shortest path. Finally, to calculate cost the cost vector and the binary vector are multiplied together. 

![Graph to illustrate the minimum cost for the shortest path in a 5 by 5 matrix](minACPP_graph.png)


[References](#References)

[Adam N. Elmachtoub, Paul Grigas (2021) Smart “Predict, then Optimize”. Management Science](https://doi.org/10.1287/mnsc.2020.3922)
[Data-Driven Conditional Robust Optimization](https://proceedings.neurips.cc/paper_files/paper/2022/file/3df874367ce2c43891aab1ab23ae6959-Paper-Conference.pdf)
