```plantuml
    skinparam componentStyle rectangle
    'skinparam linetype ortho

    ' Define components and connections
    component "Estimator_Likelihood" #LightBlue {
        component "Estimator" as est #Pink
        component "PDV" as PDV #Pink

        est -r-> PDV : A, r
    }

    ' Define invisible nodes for inputs and outputs
    () "u, z" as AR
    () pdv
    () setup as " "

    ' Input connections
    est <-l- AR

    setup -d-> est : λ, k, b, dt, H, Q, R, x0, noisy

    ' Output connections
    PDV -> pdv

```
