```plantuml
    skinparam componentStyle rectangle
    'skinparam linetype ortho

    ' Define components and connections
    component "Joint_Probability" #LightBlue {
        component "Conditional_Probability_Update" as CPU #Pink
        component "Weighted_Estimate" as WE #Pink

        CPU -r-> WE : "pdv(1, k)(t0, tn)"
        CPU -r-> CPU
    }

    ' Define invisible nodes for inputs and outputs
    () "pdv(1, k)(ti)" as PDV
    () "λ_hat" as λ
    () λs

    ' Input connections
    CPU <-l- PDV

    λs -> CPU
    λs -d--> WE

    ' Output connections
    WE -> λ

```
