```plantuml
    skinparam componentStyle rectangle
    skinparam linetype ortho

    () setup1 as " "
    () setup2 as " "
    () "S(λ)"

    component "Model" #LightBlue {
        Component [System_Dynamics] #Pink {
        
        }
    }
    
    setup1 --d--> Model : λ, k, b, dt
    
    setup2 --d--> [System_Dynamics] : H, Q, R

    Model -d-> "S(λ)"

```
