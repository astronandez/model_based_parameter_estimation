```plantuml
    skinparam componentStyle rectangle
    'skinparam linetype ortho

    ' Define components and connections
    component "MMAE" #LightBlue {
        collections "Estimator_Likelihood" as EL #Pink
        component "Joint_Probability" as JP #Pink

        EL -> JP : pdv(1, k)(ti)
    }

    ' Define invisible nodes for inputs and outputs
    () "λ_hat" as λ
    () λs
    () "u, z" as u
    () setup as " "

    ' Input connections
    EL <-l- u
    setup -d-> EL : λ_k, k, b, dt, H, Q, R, x0, noisy
    note "Each Estimator_Likelihood instance\nis passed in its own λ from λs.\nAll other inputs are the same." as N1
    N1 <- setup

    λs -d-> JP

    JP -> λ


```
