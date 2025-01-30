```plantuml
    skinparam componentStyle rectangle
    'skinparam linetype ortho

    () setup1 as " "
    () setup2 as " "
    () "S(λ)"
    () "u, x̂_ (or None)"
    () "x, z"

    Component "System_Simulator" #LightBlue {
        Component "Model" #Pink {
            
        }

        Component "Plant" #Pink {
            
        }
        
    }
    
    setup1 -d-> Model : λ, k, b, dt, H, Q, R

    setup2 --d-> Plant : x0, noisy

    "u, x̂_ (or None)" -r-> Plant

    Model -d-> Plant : "S(λ)"
    Model --d-> "S(λ)"

    Plant --r--> "x, z"

```
