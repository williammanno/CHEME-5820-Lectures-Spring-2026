L3c Notes

Binary Classification Problem

Linear Regression can be adapted for classification by transforming continuous
output into class designation

Perceptron 
- Maps σ:R --> {-1, 1} using σ(*) = sign(*)
- Learns linear decision boundary between two classes by repeatedly processing data
- Parameter vector θ updated until number of misclassifications meets specified threshold
- Computes yi for feature vector xi using sign: R --> {-1,1} function:
        yi = sign(xi^T, θ)
- Sign(z) = {1 if z ≥ 0; -1 if z ≤ 0}

Online Perception Training:

- Initialization:
    - Given dataset D = {(x1,y1),...,(xn,yn)}, max number of iterations T, and max number of mistakes M, inititalize θ = (w,b) to small random vaulues and set loop counter to t <-- 0
        - Rule of thumb for max iterations T: T = 10n to 100n where n is number of training examples --> algorithm converges faster for linearly separable data
    - While true:
        1. Initialize number of mistakes = 0
        2. For each training exapmle (x,y) in D: computer y(θ^Tx) ≤ 0
            - If true: example misclassified (sign of prediction doesnt match label y). Update θ <-- θ + yx and increment mistakes <-- mistakes + 1
        3. After processing all examples, if mistakes ≤ M or t ≤ T, exit. Otherwisem increment t <-- t + 1 and repeat from step 1
    - Aim to minimize mistakes with M = 0 being ideal
- Perceptron learns from mistakes
- θ stores hyperplane

Ex. of Perceptron Bank Note Dataset:
- -1 means genuine banknote and 1 means forged banknote
- Perceptron mistake percetage: 1.09%
    - were we more biased to forged or genuine banknotes?
- Error Analysis:
    - Confusion Matrix: 
        - Actual Positive & Predicted Positive = True Positive
        - Actual Positive & Predicted Negative = False Negative
        - Actual Negative & Predicted Negative = True Negative
        - Actual Negative & Predicted Positive = False Positive