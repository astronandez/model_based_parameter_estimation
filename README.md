# Parameter Estimation of Linear Dynamic System

## Introduction

This research project was proposed as a solution for General Electric (GE) and their DARPA sponsored project Automated-Domain Understanding and Collaborative Agency (ADUCA). ADUCA's goal is the unsupervised extraction of conceptual knowledge contained within images and videos for artificial intelligence (AI) gathering. One such avenue, is considering the physical phenomenon that is exhibited by the subject. While object detection strategies exist, these implementations only capture qualitative information present in the 2D medium, effectively losing most quantitative information convayed by the subject. 

To address this problem, we propose our implementation of parameter estimation using traditional Kalman Filter state estimation. By developing Model's (k) from tradiational physics equations, and converting into our Systems (S) mandated in {A, B, H, Q, R} matrix format, we can derive variations of our matrices S(λ_k) = {A(λ_k), B(λ_k), H, Q, R}. Where λ_k represents the parameter that we are interest in estimating and k represents the number of models variations of the value λ.

Then by comparing the covariance of our Kalman Filter residual, which is the difference between the observed state measurement and the predicted state. By assuming that our noise is normally distributed and gaussian such that our measurements are a product of the following relationship:

z = Hx + v , v ~ N(0, R)

x_hat_ = Ax + Bu + w , w ~ N(0, Q)

With these equations, an accurate system model S(λ), and assuming that our observed noise is additive white gaussian noise (AWGN). From these relationships, we can draw a time correlation from the filter's residuals using the Multiple Modle Adaptive Estimation (MMAE) algorithim to generate hypothisized conditional probabilities P(λ_k)(t - 1) that can be used to evaulate the conditional densities of the current sensor measurements z.[1]  

## Installation

1. Clone the repository with HTTPS or SSH :

        git clone https://git.uclalemur.com/astronandez/aduca_demo.git

2. Open the cloned project directory in a terminal window.

3. Install the reuirements.txt file:

        pip install -r requirements.txt

4. Run the demo_spring.py file to record, and generate, data captured via computer vision measurements.

5. Run the demo_MMAE.py with your python interpreter to see the results of the MMAE algorithm's estimation of the mostlikely parameter value.

# State Estimator Diagram

x' = Ax + Bu + w , w ~ N(0, Q)

z = Hx + v , v ~ N(0, R)

Linear AWGN Plant: System S (not constant). S is a function of unknown parameter λ = m (Specifically S = {A(m), B(m), H, Q, R})


```plantuml
    skinparam componentStyle rectangle

    () inputs1
    () inputs2
    () outputs

    component "Kalman Filter" #Pink {
        Component [Update] {
        
        }
        Component [Predict] {

        }
    }
    
    inputs1 --> Update : z(t)
    inputs2 --> Predict : u(t)
    inputs2 --> Predict : x̂_o, P_o
    
    Update -> Predict : x̂(t)
    Update -> Predict : P(t)
    Predict -> Update : x̂(t-1)
    Predict -> Update : P(t-1)

    Update -down->    outputs : x̂(t)
    Update -down->    outputs : P(t)

```

```plantuml
    skinparam componentStyle rectangle
    skinparam linetype ortho

    actor () "inputs"
    () "outputs"

    component "State Estimator SE(S(λ))" #LightBlue {
        
        Component "Kalman Filter" as KalmanFilter #Pink {

        }
    }

    Component "System S(λ)" as SystemS {
        
    }
    Component "Plant (x)" as PlantX {

    }
    Component "Input Generator (u)" as InputGenerator {

    }

    component [hidden1] as h1
    component [hidden2] as h2
    hide h1
    hide h2

    InputGenerator -u-> h1
    PlantX -r- h2

    "inputs" --> SystemS : λ

    SystemS ---> InputGenerator : "{A(λ), B(λ), H, Q, R}"
    SystemS ---> PlantX : "{A(λ), B(λ), H, Q, R}"
    SystemS ---> "State Estimator SE(S(λ))" : "{A(λ), B(λ), H, Q, R}"

    InputGenerator -> PlantX : u(t)
    InputGenerator -u-> KalmanFilter : u(t)

    PlantX -right-> KalmanFilter : z(t)

    KalmanFilter ---> "outputs" : x̂(t), P(t), Σ
    
    'PlantX --> [Validation] : z(t)
    'PlantX --> [Validation] : x(t)
    'InputGenerator --> [Validation] : u(t)
    'SystemS --> [Validation] : {A, B, H, Q, R}
    'KalmanFilter --> [Validation] : x̂(t)
    'KalmanFilter --> [Validation] : P(t)


```

# Parameter Estimator Diagram

**Given** a parameter λ and a linear system S(λ) -> (A(λ), B(λ), H, Q, R). **Find** a function that generates parameter estimate ~λ as a function of input u, output z **such that** MLE: ~λ = argmax(p(z|u, λ)).

```plantuml
    skinparam componentStyle rectangle
    skinparam linetype ortho

    () "inputs1"
    () "inputs2"
    () "S(λ_1)"
    () "S(λ_2)"
    () "S(λ_n)"
    () "outputs"

    component "Parameter Estimator" #Orange {
        Component "SE(S(λ_1))" as System1 #LightBlue {
        
        }
        Component "SE(S(λ_2))" as System2 #LightBlue {

        }
        Component "SE(S(λ_n))" as System3 #LightBlue{

        }
        Component "Selector" as Selector #LimeGreen {

        }

        System1 -[hidden]-> System2
        System2 -[hidden]-> System3
    }

    "inputs1" -left-> System1 : u
    "inputs2" -left-> System1 : z
    "S(λ_1)" ----> System1 : {A(λ_1), B(λ_1), H, Q, R}

    "inputs1" -left-> System2 : u
    "inputs2" -left-> System2 : z
    "S(λ_2)" ----> System2 : {A(λ_2), B(λ_2), H, Q, R}

    "inputs1" -left-> System3 : u
    "inputs2" -left-> System3 : z
    "S(λ_n)" ----> System3 : {A(λ_n), B(λ_n), H, Q, R}


    System1 -right-> Selector : Σ^_1
    System2 -right-> Selector : Σ^_2
    System3 -right-> Selector : Σ^_n

    Selector --> "outputs" : ~λ


```

# Selector (MMAE) Diagram

Within this block, the "Likelihood" section calculates the likelihood L(λi) of each model given the current measurement. These likelihoods are then used in the "Probability Update" section, where Bayes' theorem is applied to update the probabilities p(λi) of each model, reflecting how likely each model is after considering the new measurement. The model with the highest posterior probability p(λi) is identified, and its associated parameter ∼λ (e.g., mass) is output as the most probable estimate. This iterative process results in a refined parameter estimation that converges towards the true system state over time.

```plantuml
    skinparam componentStyle rectangle
    skinparam linetype ortho

    () "z(t)" as z
    () "inputs_1"
    () "inputs_2"
    () "inputs_n"
    () "output"

    component "Selector (MMAE)" #LimeGreen {
        Component "Likelihood(λ_1)" as L1 {
            
        }
        Component "Likelihood(λ_2)" as L2 {

        }
        Component "Likelihood(λ_n)" as Ln {

        }
        Component "Probability Update" as PU {
            
        }
        Component "max(p(λ_i))" as sel {
            
        }
        note right of PU : Bayes Theorem

    }

    z -r-> L1
    z -r-> L2
    z -r-> Ln

    inputs_1 ---> L1 : x̂(t)
    inputs_1 ---> L1 : P(t)

    inputs_2 ---> L2 : x̂(t)
    inputs_2 ---> L2 : P(t)

    inputs_n ---> Ln : x̂(t)
    inputs_n ---> Ln : P(t)

    L1 -d-> PU : L(λ_1)
    L2 -d-> PU : L(λ_2)
    Ln -d-> PU : L(λ_n)

    PU -d-> sel : p(λ_1, λ_2,.. λ_n)

    sel --> output : ~λ
   
```

# References

    [1] P. D. Hanlon and P. S. Maybeck, "Multiple-model adaptive estimation using a residual correlation Kalman filter bank," in IEEE Transactions on Aerospace and Electronic Systems, vol. 36, no. 2, pp. 393-406, April 2000, doi: 10.1109/7.845216