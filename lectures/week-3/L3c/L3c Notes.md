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


Logistic Regression:
- Adopt probabilistic framework for binary classification
- For any state y (y between {-1, 1}) with energy E(y, x^) at inverse temperature beta, conditional probability of observing the label y in {-1, +1} given the feature vector x^ can be represented as (see lecture for P(y|x^))
- Z(x^) is a partition function that normalizes the probabilities
- β > 0 is fundamental parameter and is the inverse temperature that controls how energy function translates into proabilities. It is the confidence or commitment of the model to its decisions
    - Large β: Exponential function becomes increasingly selective, small energy differences translate into large probability differences --> high confidence model eventually resembling sign function of perceptron
    - Small β: Exponential function flattens, all states become roughly equally probable regardless of energy differences. Uncertain model assigning probaility near 0.5 to both classes
    - β = 1 (default): Provides balance between confidence and uncertainty, used as baseline

- Energy Function: E(y,x^) = -y(x^T*θ)

- Use gradient descent to minimize negative log-likelihood (known as cross-entropy loss function)
    