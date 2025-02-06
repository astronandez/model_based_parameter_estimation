# Title: Computer Vision Informed Parameter Estimation

## Summary:

Our approach utilizes quantitative analysis of linear dynamic systems and modern computer vision (CV) techniques, to enrich our understanding of objects of interest in a time-series dataset. We accomplish this by generating estimates of intrinsic model parameters, by correlating simple physics-based models with time-series data subject to additive white gaussian noise. This enables us to provide users with a higher level of understanding of the object’s dynamic properties and refine our qualitative analysis.

## Splash images

TBD

## Project git repo(s):

https://github.com/astronandez/model\_based\_parameter\_estimation

# Big picture

## What is the overall problem that this and related research is trying to solve?

Provided a time-series dataset of a system in motion, and an accurate linear dynamic model of said system, how might we discern accurate estimates of the system’s intrinsic model parameters, and what do these new insights tell us about the system itself?

## Why should people (everyone) care about the problem?

Traditional methods of obtaining estimates of intrinsic parameters like mass, elasticity, and/or thermodynamics typically require specialized sensors that often necessitate considerable effort to reach the desired scale or are difficult to incorporate into everyday scenarios. Approaches like object detection, tracking, and segmentation, have received a plethora of attention over the last decade and enable developers to collect a large quantity of data using a simple camera system. Even still, these methods primarily rely on qualitative features contained in a series of images to dictate identification, leaving the quantitative information effectively unutilized. Rather than rely on qualitative features to characterize broad object categories, we can instead, identify an object using those features, and then refine our classification based on its physical behavior over the history of its time-series dataset. By capturing the object's quantitative data, we may be able to provide insightful answers to targeted questions regarding object behavior that require a deeper understanding of the object’s composition. Questions like; is the ball filled with enough air? How many passengers are in the vehicle? or how hard was the object thrown? may be questions that are answerable by considering both the qualitative and quantitative data relayed in a time-series dataset. 

## What has been done so far to address this problem?

Several advancements in CV have dramatically improved the granularity of detections at both the classification and bounding box levels. These improvements have enabled developers to create unique solutions using CV to estimate some intrinsic model parameters, like mass, but are limited by their application scope. Limited research has been done in the cross-section of CV and parameter estimation, and has primarily focused on estimating physical object location behind occlusion and image transformation properties. 

# Specific project scope

## What subset of the overall big picture problem are you addressing in particular?

This research project was proposed as a solution for General Electric (GE) and their DARPA sponsored project Automated-Domain Understanding and Collaborative Agency (ADUCA). ADUCA's goal is the unsupervised extraction of conceptual knowledge contained within images and videos for artificial intelligence (AI) gathering. One such avenue is considering the physical phenomenon that is exhibited by the subject. While object detection strategies exist, these implementations only capture qualitative information present in the 2D medium, effectively losing most quantitative information conveyed by the subject.

To address this problem, we propose our implementation of parameter estimation using traditional Kalman Filter state estimation. By developing Model's ($k$) from traditional physics equations, and converting into our Systems ($S$) mandated in $\{A, B, H, Q, R\}$ matrix format, we can derive variations of our matrices $S(\lambda_k) = \{A(\lambda_k), B(\lambda_k), H, Q, R\}$. Where $\lambda_k$ represents the parameter that we are interested in estimating and k represents the number of model variations of the value λ.

Then by comparing the covariance of our Kalman Filter residual, which is the difference between the observed state measurement and the predicted state. By assuming that our noise is normally distributed and gaussian such that our measurements are a product of the following relationship:
$$
\begin{align*}
z = &Hx + v , v ~ N(0, R) \\
\hat{x} = Ax &+ Bu + w , w ~ N(0, Q)
\end{align*}
$$

With these equations, an accurate system model $S(\lambda)$, and assuming that our observed noise is additive white gaussian noise (AWGN). From these relationships, we can draw a time correlation from the filter's residuals using the Multiple Model Adaptive Estimation (MMAE) algorithm to generate hypothesized conditional probabilities $P(\lambda_k)_{t - 1}$ that can be used to evaluate the conditional densities of the current sensor measurements z.

From this point we will conduct further research into parameter estimation techniques to optimize an algorithm for reduced error/uncertainty, efficiency, and latency of parameter estimates.

## How does solving this subproblem lead towards solving the big picture problem?

Quantitative analysis of video streams to characterize objects is the root of the problem we are aiming to solve. If we are able to find these values with a reasonably low uncertainty, then we will be able to use this information to qualitatively inform the user of potential hazards, risks, observations that could deceive even the human eye. This physics based approach gives us more tools to analyze video streams and make conclusions.

## What is your specific approach to solving this subproblem?

**State Estimator Diagram**
$$
\begin{align*}
x' = Ax + Bu &+ w \quad w \sim N(0, Q) \\
z = Hx + &v \quad v \sim  N(0, R)
\end{align*}
$$

Linear AWGN Plant: System $S$ (not constant). Where $S$ is a function of unknown parameter $\lambda = m$, more specifically:

$$
\begin{align*}
S = \{A(m), B(m), H, Q, R\}
\end{align*}
$$

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
      
    'PlantX \--\> \[Validation\] : z(t)  
    'PlantX \--\> \[Validation\] : x(t)  
    'InputGenerator \--\> \[Validation\] : u(t)  
    'SystemS \--\> \[Validation\] : {A, B, H, Q, R}  
    'KalmanFilter \--\> \[Validation\] : x̂(t)  
    'KalmanFilter \--\> \[Validation\] : P(t)
```

**Parameter Estimator Diagram**

**Given** a parameter $\lambda$ and a linear system $S(\lambda) = \{A(\lambda), B(\lambda), H, Q, R\}$. **Find** a function that generates parameter estimate ~$\lambda$ as a function of input u, output z **such that** $MLE$: ~$\lambda = argmax(p(z|u, \lambda))$.

```plantuml  
    skinparam componentStyle rectangle  
    skinparam linetype ortho

    () "z"  
    () "u"  
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

    "u" -left-> System1 : u  
    "z" -left-> System1 : z  
    "S(λ_1)" ----> System1 : {A(λ_1), B(λ_1), H, Q, R}

    "u" -left-> System2 : u  
    "z" -left-> System2 : z  
    "S(λ_2)" ----> System2 : {A(λ_2), B(λ_2), H, Q, R}

    "u" -left-> System3 : u  
    "z" -left-> System3 : z  
    "S(λ_n)" ----> System3 : {A(λ_n), B(λ_n), H, Q, R}

    System1 -right-> Selector : Σ^_1  
    System2 -right-> Selector : Σ^_2  
    System3 -right-> Selector : Σ^_n

    Selector --> "outputs" : ~λ

```

**Selector (MMAE) Diagram**

Within this block, the "Likelihood" section calculates the likelihood $L(\lambda_i)$ of each model given the current measurement. These likelihoods are then used in the "Probability Update" section, where Bayes' theorem is applied to update the probabilities $p(\lambda_i)$ of each model, reflecting how likely each model is after considering the new measurement. The model with the highest posterior probability $p(\lambda_i)$ is identified, and its associated parameter ~$\lambda$ (e.g., mass) is output as the most probable estimate. This iterative process results in a refined parameter estimation that converges towards the true system state over time.

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

This is the current implementation. We will need to reimplement the selector box with an optimized algorithm taking into account recent research on parameter estimation to find the optimized parameter estimate with the least amount of error and uncertainty. Additionally, it will need to be computationally efficient and have a low latency so it can be processed in real-time from a video stream.

## How can you be reasonably sure this approach will result in a solution?

We are confident in our approach, as we’ve had success with simulated physics-based experiments that demonstrate that under perfect conditions our algorithm can estimate the simulated model parameter values with a high degree of accuracy and precision.

In our testing we will also compare against baseline performance levels (ML algorithms for mass estimation, vehicle bridge scales, etc). The goal is to estimate the true parameter set in the system with less uncertainty than a provided baseline performance. If we can do this through modeling system noise levels accurately then we can validate our approach as a reasonable solution.

## How will we know that this subproblem has been satisfactorily solved, using quantitative metrics?

We can reasonably assume that our approach is a viable solution by comparing our results to a real-world system contained in our model corpus. In this paper, we focus on car class and derive our baseline from a web-scraped dataset, containing over 100,000 unique entries, which was later processed to find the average mass of each car class. We then compare these baseline guesses and our estimated parameter of mass against the manufacturer-rated curb weight and various known loads to calculate our metrics of accuracy and precision. Through this comparison, we will be able to demonstrate that our approach is a better estimator than informed guessing.

If we model baseline performance error for predicting the true parameter set as $\epsilon b$ and the same metric for our implementation as $\epsilon a$ we aim to find that argmin($\epsilon a$, $\epsilon b$) \= $\epsilon a$ by at least 10%. E.g. $\epsilon a$ < 0.9 $\epsilon b$ 

# Broader impact

## What is the value of your approach beyond this specific solution?

While our research initially concentrates on estimating attributes of vehicles in video streams, the methodologies we have developed are versatile and can be extended to various contexts involving different objects within video footage. Our approach is particularly valuable for analyzing physical systems with undetermined parameters that can be described through mathematical models. By estimating these parameters directly from video data, we significantly enhance our ability to gather insights about the object in question. This not only broadens our understanding of the object's affordances but also opens new avenues for in-depth analysis, making our method a powerful tool for a wide range of applications.

## What is the value of this solution beyond solely solving this subproblem and getting us closer to solving the big picture problem?

Our solution can eventually be applied to vehicle traffic systems, navigation systems, guidance systems, and many others to assess safety, hazards, and objectives given video streams. Additionally, this technique enables the quantitative analysis of temporal attributes of objects in video streams to infer their characteristics. It can be applied to a wide range of real-world systems, provided that a linear model can be established for the object's dynamics.

## Background / related work / references

[https://github.com/astronandez/model\_based\_parameter\_estimation/tree/main/docs](https://github.com/astronandez/model_based_parameter_estimation/tree/main/docs)

## System capabilities, validation deliverables, engineering tasks

## Concrete external deadlines (paper submissions):

ICCV Paper Registration Deadline: 	March 7th, 2025 11:59pm HST

## Detailed schedule (weekly capabilities / deliverables / tasks):

| Date | Capabilities | Tasks | Deliverables |
| :---: | ----- | ----- | ----- |
| 01/30 | All systems running in GitHub repo | Upload remaining scripts from the previous code bases Set up new experiment | Repo Link Project Proposal w/o Splash Image Video of new experiment set up |
| 02/06 | Ability to evaluate a 2d model of a spring mass damper system Try with 1d as well Tentative: Ability to estimate between parameter sets with a high precision Validate pipeline integrity Ability to record and measure from sticky note videos | Evaluation of results for noise conditions from last quarter Additionally for new 2d model Record new spring videos with experiment setup and measurements Run evaluation in new repo Look for the spring model in textbook for citation IF ABLE: Film videos of multiple sizes of car (sedan, suv, truck) and differing numbers of people (1-4) | Initial 2d model with Parameter Estimation Graphs (heatmap likelihoods, estimations over time) Completed powerpoint slides showing results for differing simulation noise combinations both with and without the true parameters in the test set |
| 02/13 | Tentative: Ability to estimate between parameter sets with a high precision Ability to assess spring experiment parameter estimation in both accuracy and precision of system parameters | Finish experiments with spring IF NECESSARY setup and record further spring experiments Begin filming videos of multiple sizes of car (sedan, suv, truck) and differing numbers of people (1-4) for test | Videos of measurements on spring mass system Slides showing validation of pipeline results vs baseline Parameter Estimation graphs (heatmap likelihoods, estimations over time) |
| 02/20 | Ability to assess car experiment parameter estimation in both accuracy and precision of system parameters | Commence first draft of paper Film videos of multiple sizes of car (sedan, suv, truck) and differing numbers of people (1-4) Test pipeline using the generated datasets and provide results and evaluation Evaluate against baseline (scraped dataset) for accuracy comparison | Reasonable first draft of paper, incomplete results section Powerpoint slides for accuracy comparison for all vehicle tests showing pipeline vs baseline accuracy |
| 02/27 | Ability to create graphs from code base for paper. | Finalize first draft Finalize results section from first draft Continue revisions of paper Paper Registration IF ABLE set up further tests | The first draft IF NECESSARY Further slides showing validation accuracy of pipeline |
| 03/06 | Paper registration deadline Mar 03 '25 11:59 PM HST | Finalize paper If necessary/able to complete further testing (new system?) | Final paper |
| 03/07 | Paper submission deadline March 7th, 2025 11:59pm HST (\~ March 8th 2:59am PST) | Paper completed | Paper completed |

- Where does the measurements come from  
  - What are the measurements (coordinates, images, etc)?  
- What are the contributions?  
  - What is a novel? What's the focus?  
- 