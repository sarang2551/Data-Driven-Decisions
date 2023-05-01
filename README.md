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

The vector x that represents features is in $\mathbb{R}^p$ while the cost vector, c, is in $\mathbb{R}^d$. In the function above, p and d are hardcoded to be 5 and 40 respectively. The dimension for the cost vector, d, correponds to the number of edges in a 5 by 5 matrix, other dimensions can be used for a n by m matrix using the following formula: Edges = n * (m-1) + m * (n-1). First, a random matrix $\mathbb{B}^* \subseteq \mathbb{R}^{d \times p}$ is generated which encodes the parameters of the true model, where each entry, b, in $ \mathbb{B}^* $ is $b \sim \text{Bernoulli}(p)$.
This function is used to generate both in-sample training data as well as out-of-sample test data.  


[Mean Squared Error metric](#mss_metric)
[Downstream decision](#downstream_decision_exp1)
[References](#References)
[Adam N. Elmachtoub, Paul Grigas (2021) Smart “Predict, then Optimize”. Management Science](https://doi.org/10.1287/mnsc.2020.3922)
[Data-Driven Conditional Robust Optimization](https://proceedings.neurips.cc/paper_files/paper/2022/file/3df874367ce2c43891aab1ab23ae6959-Paper-Conference.pdf)
