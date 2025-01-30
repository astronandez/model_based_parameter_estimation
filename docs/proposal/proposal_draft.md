# Title:

Quantitative Analysis and Optimal Parameter Estimation in Linear Dynamic Systems

### Summary:

Our approach focuses on quantitative analysis using modern computer vision techniques, which are divided into qualitative and quantitative paths. We prioritize the latter to enhance the understanding of objects within a scene. This is achieved by finding an optimal parameter estimation of the system, which involves the estimation of linear dynamic system parameters. By correlating simple physics-based models with time-series data subject to additive white Gaussian noise, we aim to refine our analysis. This method allows us to provide users with in-depth information about the objects' dynamics, potentially enriching their understanding and analysis of affordances in the video streams.

### Splash images

Placeholder image: Marc updating the new diagram

![image info](./images/problem.png)

### Project git repo(s):

https://git.uclalemur.com/astronandez/aduca_demo

## Big picture 

### What is the overall problem that this and related research is trying to solve?

Given time series data about an object in motion and a physics-based model that describes this motion, how can we accurately estimate the object's quantitative characteristics? These estimates can then help us understand the object's capabilities and affordances.

### Why should people (everyone) care about the problem?

Solving this problem offers a transformative approach to understanding the physical capabilities and behaviors of objects by leveraging video streams. Traditionally, analyzing an object's motion or load requires the installation of specialized sensors, precisely oriented to capture the necessary data. However, many environments lack the infrastructure to support such sensor networks, whereas video streams are more readily available. For instance, utilizing street cameras enables us to employ parameter estimation techniques to deduce the mass of a vehicle. This insight can significantly enhance our predictions about a vehicle's required braking time to halt completely. If a driver fails to initiate braking within this calculated timeframe, it suggests a more aggressive driving style. This solution not only circumvents the limitations imposed by the lack of sensor infrastructure but also provides a scalable, efficient method to assess and predict behaviors in a multitude of settings.

### What has been done so far to address this problem?

Extensive research has been conducted on computer vision techniques for object detection. Additionally, there is a large body of work on parameter estimation for Linear AWGN systems dating back thirty years. Research has been conducted to characterize affordances through neural network training or by using a large knowledge base but this can be computationally expensive and also faces limitiations to what was within the original knowledge base.

## Specific project scope

### What subset of the overall big picture problem are you addressing in particular?

This research project was proposed as a solution for General Electric (GE) and their DARPA sponsored project Automated-Domain Understanding and Collaborative Agency (ADUCA). ADUCA's goal is the unsupervised extraction of conceptual knowledge contained within images and videos for artificial intelligence (AI) gathering. One such avenue, is considering the physical phenomenon that is exhibited by the subject. While object detection strategies exist, these implementations only capture qualitative information present in the 2D medium, effectively losing most quantitative information convayed by the subject.

To address this problem, we propose our implementation of parameter estimation using traditional Kalman Filter state estimation. By developing Model's (k) from tradiational physics equations, and converting into our Systems (S) mandated in {A, B, H, Q, R} matrix format, we can derive variations of our matrices S(λ_k) = {A(λ_k), B(λ_k), H, Q, R}. Where λ_k represents the parameter that we are interest in estimating and k represents the number of models variations of the value λ.

Then by comparing the covariance of our Kalman Filter residual, which is the difference between the observed state measurement and the predicted state. By assuming that our noise is normally distributed and gaussian such that our measurements are a product of the following relationship:

z = Hx + v , v ~ N(0, R)

x_hat_ = Ax + Bu + w , w ~ N(0, Q)

With these equations, an accurate system model S(λ), and assuming that our observed noise is additive white gaussian noise (AWGN). From these relationships, we can draw a time correlation from the filter's residuals using the Multiple Modle Adaptive Estimation (MMAE) algorithim to generate hypothisized conditional probabilities P(λ_k)(t - 1) that can be used to evaulate the conditional densities of the current sensor measurements z.

From this point we will conduct further research into parameter estimation techniques to optimize an algorithm for reduced error/uncertainty, effiency, and latency of parameter estimates.

### How does solving this subproblem lead towards solving the big picture problem?

Quantitative analysis of video streams to characterize objects is the root of the problem we are aiming to solve. If we are able to find these values with a reasonably low uncertainty, then we will be able to use this information to qualitatively inform the user of potential hazards, risks, observations that could deceive even the human eye. This physics based approach gives us more tools to analyze video streams and make conclusions.

### What is your specific approach to solving this subproblem?

**State Estimator Diagram**

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

**Parameter Estimator Diagram**

**Given** a parameter λ and a linear system S(λ) -> (A(λ), B(λ), H, Q, R). **Find** a function that generates parameter estimate ~λ as a function of input u, output z **such that** MLE: ~λ = argmax(p(z|u, λ)).

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

This is the current implementation. We will need to reimplement the selector box with an optimized algorithm taking into account recent research on parameter estimation to find the optimized parameter estimate with the least aount of error and uncertainty. Additionally, it will need to be computationally efficient and have a low latency so it can be processed in real-time from a video stream.


### How can you be reasonably sure this approach will result in a solution?

The accuracy of the approach and solution will be based upon the estimation error metrics.

### How will we know that this subproblem has been satisfactorily solved, using quantitative metrics?

As mentioned above, we will need to analyze the error metrics to validate the approach. We will need to prove that our ouput parameter esimation is the argmax p(z|u, λ) and has the lowest uncertainty of all possible outcomes.

## Broader impact
(even if someone doesn't care about the big picture problem that you started with, why should they still care about the specific work that you've produced?  Who else can use your processes and results, and how?)

### What is the value of your approach beyond this specific solution?

While our research initially concentrates on estimating attributes of vehicles in video streams, the methodologies we have developed are versatile and can be extended to various contexts involving different objects within video footage. Our approach is particularly valuable for analyzing physical systems with undetermined parameters that can be described through mathematical models. By estimating these parameters directly from video data, we significantly enhance our ability to gather insights about the object in question. This not only broadens our understanding of the object's affordances but also opens new avenues for in-depth analysis, making our method a powerful tool for a wide range of applications.

### What is the value of this solution beyond solely solving this subproblem and getting us closer to solving the big picture problem?

Our solutions can eventually be applied to vehicle traffic systems, naviagtion systems, guidance systems, and many others to assess safety, hazards, and objectives given video streams.

## Background / related work / references

TODO - Link to your literature review in your repo.

## System capabilities, validation deliverables, engineering tasks

### Concrete external deadlines (paper submissions):

Conference - TBD (previously ICRA 09/24)

Title - Quantitative Analysis and Optimal Multi-Parameter Estimation in Linear Dynamic Systems

Abstract - TODO

### Detailed schedule (weekly capabilities / deliverables / tasks):

| Start Date         | Goals                | Tasks                                                        | Deliverables                                                 |
| ------------ | -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 10/03 | <ul><li>Planning</li><li>Re-familiarizing w/previous work</li><ul> | <ul><li>Measure Spring Constants</li><li>Re-familiarize self with work completed last year</li><ul> | <ul><li>Finalize Project Proposal</li><li>Finalize project schedule/calendar for completion by Nov. 1st</li><ul> |
| 10/04 | <ul><li>Confirm CVMMAE function state  </li><li>Confirm validation of MMAE simulator from Spring</li><ul> | <ul><li>Set up Spring Mass Damper Testing Rig and Camera </li><li>Validate successful evaluation using monoparameter estimator on different rig configurations</li><li>Modiy graph interaces as needed</li><li>Validate successful completion of bug-fixing for the MMAE simulator from last quarter (Confirm successful testbenches)</li><ul> | <ul><li>Confirm proposal and schedule with Peter (ADUCA group)</li><li>Finalize function for Spring constant monoparameter as missing parameter</li><li>Initial Plot of monoparameter estimation against time </li><ul> |
| 10/07 | <ul><li>Planning multivariable estimator</li><li>Complete Graph interfaces for evaluations using monoparameter</li><ul> | <ul><li>CV evaluations using monoparameter estimation on spring mass for each parameter type</li><li>Research on multivariable parameter estimator techniques for implementation w/in our current MMAE pipeline</li><ul> | <ul><li>Plot of monoparameter estimates over time for each type of parameter value</li><li>Document summarizing the research found on multivariable parameter estimation and thoughts on how to implement this w/in our current single variable parameter estimator</li><ul> |
| 10/09 | <ul><li>Upload smaple videos of nonoparameter estimation of each parameter type with graph and overlay</li><li>Planning multivariable estimator</li><ul> | <ul><li>Develop block diagram for multivariable estimator that builds on current MMAE pipeline</li><li>Finish spring test monoparameter evaluations</li><li>Graph the level of confidence from our spring test results</li><ul> | <ul><li>Block diagrams of multivariable estimator design </li><li> Scatter plot with Confidance Intervals and error metric labels for monoparameter estimation</li><ul> |
| 10/11 | <ul><li>Update multivariable estimator plan</li><li>Upload results of monoparameter estimation on remaining spring mass tests experiments</li><ul> | <ul><li>Based upon feedback on multivariable estimator block diagram design/research, update block diagram so code integration will mirror block diagram exactly</li><li> Document spring test experiment proceedure and revise any block diagram changes</li><ul> | <ul><li>Final block diagram of multivariable estimaor design to combine with the current MMAE pipeline</li><li>Test proceedure documentation and final block diagrams for CV components and integration</li><ul> |
| 10/14 | <ul><li>Create Vehicle model (ABHQR) matricies</li><li>Recover vehicle computer vision model</li><li>Multivariable parameter estimator code implementation</li><ul> | <ul><li>Create new .json config files</li><li>Take photos of test vehicles for model training and find vehicle baselines</li><li>Based upon the finalized multivariable parameter estimator block diagram, implement the design in the code to exactly mirror the block diagram (floowing the design from last quarter i.e. class for each block, update function for each clas etc.)</li><ul> | <ul><li>Documentation and code on vehicle model and source</li><li>Preliminary code implementation of the multivariable parameter estimator</li><ul> |
| 10/16 | <ul><li>Retrain vehicle CV model and upload LEMUR videos</li><li>Validate multivariable estimator code against block diagram, begin testing</li><ul> | <ul><li>Record LEMUR videos going over speed bumps</li><li>Upload Dataset to Roboflow</li><li> Select ADUCA videos</li><li>Validate multivariable estimator code against block diagram, begin testing</li><ul> | <ul><li>LEMUR and ADUCA video recordings and working vehicle CV system</li><li>Slides of block diagram - code implementation validating they're identical. Initial tests (plots conveying initial success)</li><ul> |
| 10/18 | <ul><li>Finalize vehicle overlay and graph for CV evaluation</li><li>Multivariable estimator testing</li><ul> | <ul><li>Update vehicle overlay </li><li>Thoroughly test multivariable estimator in simulation</li><ul> | <ul><li>LEMUR video with preliminary overlay sheme</li><li>Tentative validation metrics and errors against baseline performance</li><li>Run monte-carlo simulation and generate heat map for different configurations</li><ul> |
| 10/21 | <ul><li>Upload LEMUR and ADUCA videos with monoparameter results</li><li>Multivariable estimator testing</li><ul> | <ul><li>CV evaluation of LEMUR and ADUCA videos using monoparameter estimation</li><li>Thoroughly test multivariable estimator in simulation</li><ul> | <ul><li>Scatter plot with Confidance Intervals and error metric labels for monoparameter estimation on LEMUR and ADUCA videos</li><li>Tentative validation metrics and errors against baseline performance</li><li>Run monte-carlo simulation and generate heat map for different configurations</li><ul> |
| 10/23 | <ul><li>Successful test of multiparameter estimation using spring mass damper system</li><li>Multivariable estimator integration testing</li><ul> | <ul><li>Preliminary testing on spring mass damper system using multiparameter estimation</li><li>Thoroughly test multivariable estimator in simulation</li><li>Integrate multivariable estimator into CV pipeline for testing on videos of spring-mass system</li><ul> | <ul><li>Plot multiparameter estimation over time for spring mass preliminary test</li><li>Videos of integration testing on the spring-mass system</li><ul> |
| 10/25 | <ul><li>Upload samples of spring test multiparameter estimations with graph and overlay</li><li>Multivariable estimator integration testing</li><ul> | <ul><li>CV evaluation of spring mass using multiparameter estimation</li><li>Begin report writing, slide creation, final video editing</li><li>Integrate multivariable estimator into CV pipeline for testing on videos of custom car setup</li><ul> | <ul><li>Scatter plot with Confidance Intervals and error metric labels for multiparameter estimation on spring mass system</li><li>Drafts of final reports and deliverables</li><li>Videos of integration testing on the spring-mass system</li><ul> |
| 10/28 | <ul><li>Upload remaing spring mass multiparameter videos</li><li>Upload LEMUR and ADUCA videos evaluations</li><ul> | <ul><li>CV evaluation of LEMUR and ADUCA videos using multiparameter estimation</li><li>Document LEMUR and ADUCA video results using multiparameter estimation</li><li>Continue report writing, slide creation, final video editing</li><ul> | <ul><li>Scatter plot with Confidance Intervals and error metric labels for multiparameter estimation on LEMUR and ADUCA datasets</li><li>Vehicle test proceedure documentation</li><li>Finalized drafts of final reports and deliverables showing the baseline performance and our systems increased accuracy on the same test data</li><ul> |
| 10/30 | <ul><li>Extra Time</li><ul> | <ul><li>Finish leftover items</li><ul> | <ul><li>Finalize deliverables</li><ul> |