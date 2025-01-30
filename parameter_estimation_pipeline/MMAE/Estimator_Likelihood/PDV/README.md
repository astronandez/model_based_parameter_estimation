```plantuml
    skinparam componentStyle rectangle
    skinparam linetype ortho

    ' Define components and connections
    component "PDV" #LightBlue {
        component "Compute_Scalar_Likelihood" as CSL #Pink
        component "Compute_Model_PDV" as CMPDV #Pink
        CSL -> CMPDV : q
    }

    ' Define invisible nodes for inputs and outputs
    () "A, r" as AR
    () pdv

    ' Input connections
    CSL <-l- AR
    CMPDV <-l- AR : A

    ' Output connections
    CMPDV -> pdv

```
