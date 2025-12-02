# Mathematics-for-Robotics-and-Machine-Learning

## 🎯 Foundation: Algebra & Pre-Calculus

### Linear Equations
- [ ] **Equation**: `ax + b = 0`
- [ ] **Application**: Solving for joint positions

### Quadratic Formula
- [ ] **Equation**: `x = (-b ± √(b²-4ac))/2a`
- [ ] **Application**: Trajectory parabolas

### Exponentials
- [ ] **Equation**: `e^x`, `a^x`
- [ ] **Application**: Discount factors `γ^t`

### Logarithms
- [ ] **Equation**: `log(xy) = log(x) + log(y)`
- [ ] **Application**: Log-likelihood optimization

### Trigonometry
- [ ] **Equation**: `sin²θ + cos²θ = 1`
- [ ] **Application**: Rotation matrices

---

## 📐 Linear Algebra (Level 1)

### Vector Operations
- [ ] **Vector addition**: `v + w = [v₁+w₁, v₂+w₂, ...]`
- [ ] **Application**: State combinations

### Scalar Multiplication
- [ ] **Equation**: `cv = [cv₁, cv₂, ...]`
- [ ] **Application**: Scaling actions

### Dot Product
- [ ] **Equation**: `v·w = Σvᵢwᵢ`
- [ ] **Application**: Neuron activation

### Matrix Multiplication
- [ ] **Equation**: `(AB)ᵢⱼ = Σₖ AᵢₖBₖⱼ`
- [ ] **Application**: Layer transformations

### Matrix Transpose
- [ ] **Equation**: `(Aᵀ)ᵢⱼ = Aⱼᵢ`
- [ ] **Application**: Backpropagation gradients

### Identity Matrix
- [ ] **Equation**: `Iv = v`
- [ ] **Application**: No transformation

### Matrix Inverse
- [ ] **Equation**: `AA⁻¹ = I`
- [ ] **Application**: Inverse kinematics

---

## 📈 Calculus (Single Variable)

### Limits
- [ ] **Equation**: `lim(x→a) f(x)`
- [ ] **Application**: Convergence checks

### Derivatives
- [ ] **Equation**: `f'(x) = lim(h→0) [f(x+h)-f(x)]/h`
- [ ] **Application**: Gradient computation

### Power Rule
- [ ] **Equation**: `d/dx(xⁿ) = nxⁿ⁻¹`
- [ ] **Application**: Polynomial derivatives

### Chain Rule
- [ ] **Equation**: `d/dx[f(g(x))] = f'(g(x))·g'(x)`
- [ ] **Application**: Backpropagation

### Product Rule
- [ ] **Equation**: `d/dx[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)`
- [ ] **Application**: Complex derivations

### Integration
- [ ] **Equation**: `∫f(x)dx`
- [ ] **Application**: Cumulative reward

### Fundamental Theorem of Calculus
- [ ] **Equation**: `∫ₐᵇ f'(x)dx = f(b) - f(a)`
- [ ] **Application**: Total change calculation

---

## 📐 Linear Algebra (Level 2)

### Determinant
- [ ] **Equation**: `det(A) = ad - bc` (2×2 case)
- [ ] **Application**: Area scaling, invertibility

### Eigenvalues and Eigenvectors
- [ ] **Equation**: `Av = λv`
- [ ] **Application**: Stability analysis

### Characteristic Equation
- [ ] **Equation**: `det(A - λI) = 0`
- [ ] **Application**: Finding eigenvalues

### Singular Value Decomposition (SVD)
- [ ] **Equation**: `A = UΣVᵀ`
- [ ] **Application**: Dimensionality reduction

### Frobenius Norm
- [ ] **Equation**: `||A||_F = √(Σᵢⱼ aᵢⱼ²)`
- [ ] **Application**: Matrix distance

### Vector Norm
- [ ] **Equation**: `||v|| = √(Σvᵢ²)`
- [ ] **Application**: Euclidean distance

---

## 🎲 Multivariable Calculus

### Partial Derivatives
- [ ] **Equation**: `∂f/∂xᵢ`
- [ ] **Application**: Gradient components

### Gradient Vector
- [ ] **Equation**: `∇f = [∂f/∂x₁, ..., ∂f/∂xₙ]`
- [ ] **Application**: Steepest ascent direction

### Chain Rule (Multivariate)
- [ ] **Equation**: `∂z/∂x = (∂z/∂y)(∂y/∂x)`
- [ ] **Application**: Neural network gradients

### Jacobian Matrix
- [ ] **Equation**: `Jᵢⱼ = ∂fᵢ/∂xⱼ`
- [ ] **Application**: Robot velocity relationships

### Hessian Matrix
- [ ] **Equation**: `Hᵢⱼ = ∂²f/∂xᵢ∂xⱼ`
- [ ] **Application**: Curvature information

### Directional Derivative
- [ ] **Equation**: `D_v f = ∇f·v`
- [ ] **Application**: Gradient in direction

### Double Integrals
- [ ] **Equation**: `∫∫f(x,y)dxdy`
- [ ] **Application**: 2D probability mass

---

## 🔄 Differential Equations

### First-Order ODEs
- [ ] **Equation**: `dy/dt = f(t,y)`
- [ ] **Application**: Velocity from acceleration

### Second-Order ODEs
- [ ] **Equation**: `d²y/dt² = f(t,y,dy/dt)`
- [ ] **Application**: Newton's second law

### Linear ODEs
- [ ] **Equation**: `dy/dt + p(t)y = g(t)`
- [ ] **Application**: Damped systems

### Exponential Solutions
- [ ] **Equation**: `y(t) = y₀e^(kt)`
- [ ] **Application**: Growth/decay models

### Euler Method
- [ ] **Equation**: `yₙ₊₁ = yₙ + h·f(tₙ,yₙ)`
- [ ] **Application**: Basic simulation step

### Runge-Kutta 4th Order (RK4)
- [ ] **Equation**: Complex 4-stage formula
- [ ] **Application**: Accurate physics simulation

### Stability Condition
- [ ] **Equation**: `Re(λ) < 0`
- [ ] **Application**: System convergence

---

## 🎲 Probability (Level 1)

### Probability Axioms
- [ ] **Equation**: `0 ≤ P(A) ≤ 1`, `P(Ω) = 1`
- [ ] **Application**: Valid probability measures

### Addition Rule
- [ ] **Equation**: `P(A∪B) = P(A) + P(B) - P(A∩B)`
- [ ] **Application**: Union probability

### Conditional Probability
- [ ] **Equation**: `P(A|B) = P(A∩B)/P(B)`
- [ ] **Application**: Bayesian updates

### Independence
- [ ] **Equation**: `P(A∩B) = P(A)P(B)`
- [ ] **Application**: Feature independence assumption

### Bayes' Theorem
- [ ] **Equation**: `P(A|B) = P(B|A)P(A)/P(B)`
- [ ] **Application**: Posterior estimation

### Law of Total Probability
- [ ] **Equation**: `P(A) = ΣP(A|Bᵢ)P(Bᵢ)`
- [ ] **Application**: Marginalization

---

## 📊 Statistics (Level 1)

### Expectation
- [ ] **Equation**: `E[X] = Σx·P(X=x)` or `∫x·f(x)dx`
- [ ] **Application**: Mean reward

### Variance
- [ ] **Equation**: `Var(X) = E[(X-μ)²] = E[X²] - E[X]²`
- [ ] **Application**: Uncertainty measure

### Standard Deviation
- [ ] **Equation**: `σ = √Var(X)`
- [ ] **Application**: Spread measure

### Gaussian (Normal) Distribution
- [ ] **Equation**: `f(x) = (1/√(2πσ²))e^(-(x-μ)²/2σ²)`
- [ ] **Application**: Noise modeling

### Uniform Distribution
- [ ] **Equation**: `f(x) = 1/(b-a)` for `x∈[a,b]`
- [ ] **Application**: Random exploration

### Law of Large Numbers
- [ ] **Equation**: `X̄ₙ → μ` as `n→∞`
- [ ] **Application**: Sample average converges

### Central Limit Theorem
- [ ] **Equation**: `√n(X̄ₙ-μ)/σ → N(0,1)`
- [ ] **Application**: Sampling distributions

---

## 🎲 Probability (Level 2)

### Joint Probability Density
- [ ] **Equation**: `f(x,y)`
- [ ] **Application**: Multi-sensor measurements

### Marginal Distribution
- [ ] **Equation**: `f_X(x) = ∫f(x,y)dy`
- [ ] **Application**: Integrating out variables

### Covariance
- [ ] **Equation**: `Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)]`
- [ ] **Application**: Correlation measure

### Correlation Coefficient
- [ ] **Equation**: `ρ = Cov(X,Y)/(σₓσᵧ)`
- [ ] **Application**: Normalized correlation

### Multivariate Gaussian
- [ ] **Equation**: `f(x) = (2π)^(-n/2)|Σ|^(-1/2)e^(-½(x-μ)ᵀΣ⁻¹(x-μ))`
- [ ] **Application**: State uncertainty representation

### Conditional Independence
- [ ] **Equation**: `P(A|B,C) = P(A|C)`
- [ ] **Application**: Graphical model simplification

---

## 📊 Statistics (Level 2)

### Likelihood Function
- [ ] **Equation**: `L(θ|x) = P(x|θ)`
- [ ] **Application**: Data given parameters

### Log-Likelihood
- [ ] **Equation**: `ℓ(θ) = log L(θ) = Σlog P(xᵢ|θ)`
- [ ] **Application**: Easier optimization

### Maximum Likelihood Estimation (MLE)
- [ ] **Equation**: `θ̂ = argmax L(θ|x)`
- [ ] **Application**: Parameter estimation

### Maximum A Posteriori (MAP)
- [ ] **Equation**: `θ̂ = argmax P(θ|x) = argmax P(x|θ)P(θ)`
- [ ] **Application**: Bayesian parameter estimation

### Posterior Distribution
- [ ] **Equation**: `P(θ|x) ∝ P(x|θ)P(θ)`
- [ ] **Application**: Updated beliefs

### Confidence Intervals
- [ ] **Equation**: `[θ̂ - z*SE, θ̂ + z*SE]`
- [ ] **Application**: Uncertainty quantification

---

## 🎯 Optimization (Level 1)

### Gradient Descent
- [ ] **Equation**: `θₜ₊₁ = θₜ - α∇L(θₜ)`
- [ ] **Application**: Parameter updates

### Learning Rate Schedule
- [ ] **Equation**: `αₜ = α₀/(1+decay·t)`
- [ ] **Application**: Adaptive step size

### Momentum
- [ ] **Equation**: `vₜ = βvₜ₋₁ + ∇L(θₜ)`, `θₜ₊₁ = θₜ - αvₜ`
- [ ] **Application**: Accelerated gradient descent

### Adam Optimizer
- [ ] **Equation**: `mₜ = β₁mₜ₋₁ + (1-β₁)∇L`, `vₜ = β₂vₜ₋₁ + (1-β₂)∇L²`
- [ ] **Application**: Adaptive learning rates

### Line Search
- [ ] **Equation**: `minimize f(x + αd)` over `α`
- [ ] **Application**: Optimal step size

### Convergence Criterion
- [ ] **Equation**: `||∇L(θₜ)|| < ε`
- [ ] **Application**: Training termination

---

## 🎯 Optimization (Level 2)

### Lagrangian
- [ ] **Equation**: `ℒ(x,λ) = f(x) + Σλᵢgᵢ(x)`
- [ ] **Application**: Constrained optimization

### KKT Conditions
- [ ] **Equation**: `∇f + Σλᵢ∇gᵢ = 0`, `λᵢgᵢ = 0`
- [ ] **Application**: Optimal constrained solution

### Quadratic Programming
- [ ] **Equation**: `min ½xᵀQx + cᵀx` subject to `Ax≤b`
- [ ] **Application**: Trajectory optimization

### Newton's Method
- [ ] **Equation**: `θₜ₊₁ = θₜ - H⁻¹∇L`
- [ ] **Application**: Second-order optimization

### Convexity Condition
- [ ] **Equation**: `∇²f ≽ 0` (positive semi-definite Hessian)
- [ ] **Application**: Guaranteed global minimum

---

## 📐 Linear Algebra (Advanced)

### Inner Product
- [ ] **Equation**: `⟨u,v⟩ = uᵀv`
- [ ] **Application**: Generalized dot product

### Orthogonality
- [ ] **Equation**: `⟨u,v⟩ = 0`
- [ ] **Application**: Perpendicular vectors

### Vector Projection
- [ ] **Equation**: `proj_v(u) = (⟨u,v⟩/||v||²)v`
- [ ] **Application**: Component extraction

### Gram-Schmidt Process
- [ ] **Equation**: `vₖ = uₖ - Σⱼ₌₁ᵏ⁻¹ proj_vⱼ(uₖ)`
- [ ] **Application**: Orthonormal basis construction

### Spectral Theorem
- [ ] **Equation**: `A = QΛQᵀ` (for symmetric A)
- [ ] **Application**: Eigendecomposition

### Positive Definite Matrix
- [ ] **Equation**: `xᵀAx > 0` for all `x≠0`
- [ ] **Application**: Valid distance metrics

---

## 📡 Information Theory

### Entropy
- [ ] **Equation**: `H(X) = -ΣP(x)log P(x)`
- [ ] **Application**: Uncertainty measure

### Cross-Entropy
- [ ] **Equation**: `H(p,q) = -ΣP(x)log q(x)`
- [ ] **Application**: Classification loss function

### KL Divergence
- [ ] **Equation**: `D_KL(p||q) = ΣP(x)log[P(x)/q(x)]`
- [ ] **Application**: Policy constraints (PPO, TRPO)

### Mutual Information
- [ ] **Equation**: `I(X;Y) = H(X) - H(X|Y)`
- [ ] **Application**: Representation learning

### Fisher Information
- [ ] **Equation**: `I(θ) = E[(∂log p(x|θ)/∂θ)²]`
- [ ] **Application**: Natural gradient methods

---

## 🔄 Dynamical Systems

### State-Space Representation
- [ ] **Equation**: `ẋ = f(x,u)`, `y = h(x)`
- [ ] **Application**: Robot system modeling

### Linearization
- [ ] **Equation**: `ẋ ≈ Ax + Bu` where `A=∂f/∂x`, `B=∂f/∂u`
- [ ] **Application**: Local linear approximation

### Lyapunov Stability
- [ ] **Equation**: `V(x) > 0`, `V̇(x) < 0`
- [ ] **Application**: Stability proof

### Transfer Function
- [ ] **Equation**: `G(s) = Y(s)/U(s)`
- [ ] **Application**: Frequency domain analysis

### Controllability Matrix
- [ ] **Equation**: `C = [B AB A²B ... Aⁿ⁻¹B]`
- [ ] **Application**: Full controllability check

### Observability Matrix
- [ ] **Equation**: `O = [C; CA; CA²; ...; CAⁿ⁻¹]`
- [ ] **Application**: State estimation feasibility

---

## 🎲 Markov Processes

### Markov Property
- [ ] **Equation**: `P(sₜ₊₁|sₜ,sₜ₋₁,...) = P(sₜ₊₁|sₜ)`
- [ ] **Application**: Memoryless state transitions

### Transition Probability Matrix
- [ ] **Equation**: `P(s'|s,a)`
- [ ] **Application**: State dynamics modeling

### Stationary Distribution
- [ ] **Equation**: `πP = π`
- [ ] **Application**: Long-term behavior

### Bellman Equation
- [ ] **Equation**: `V(s) = maxₐ[R(s,a) + γΣₛ'P(s'|s,a)V(s')]`
- [ ] **Application**: Optimal value function

### Q-Function
- [ ] **Equation**: `Q(s,a) = R(s,a) + γΣₛ'P(s'|s,a)V(s')`
- [ ] **Application**: Action-value estimation

### Policy Iteration
- [ ] **Equation**: `πₖ₊₁(s) = argmaxₐ Q^πₖ(s,a)`
- [ ] **Application**: Policy improvement

---

## 🔢 Numerical Methods

### Newton-Raphson Method
- [ ] **Equation**: `xₙ₊₁ = xₙ - f(xₙ)/f'(xₙ)`
- [ ] **Application**: Root finding, inverse kinematics

### Bisection Method
- [ ] **Equation**: If `f(a)f(b)<0`, root exists in `[a,b]`
- [ ] **Application**: Robust root finding

### Linear Interpolation
- [ ] **Equation**: `y = y₀ + (y₁-y₀)(x-x₀)/(x₁-x₀)`
- [ ] **Application**: Trajectory smoothing

### Spline Interpolation
- [ ] **Equation**: Piecewise polynomials with continuity
- [ ] **Application**: Smooth path generation

### Forward Difference
- [ ] **Equation**: `f'(x) ≈ [f(x+h)-f(x)]/h`
- [ ] **Application**: Numerical derivatives

### Trapezoidal Rule
- [ ] **Equation**: `∫ₐᵇf(x)dx ≈ (h/2)[f(x₀)+2f(x₁)+...+f(xₙ)]`
- [ ] **Application**: Numerical integration

---

## 🎓 Advanced Calculus

### Functional
- [ ] **Equation**: `J[y] = ∫L(x,y,y')dx`
- [ ] **Application**: Path cost optimization

### Euler-Lagrange Equation
- [ ] **Equation**: `d/dt(∂L/∂q̇) - ∂L/∂q = 0`
- [ ] **Application**: Equations of motion

### Lagrangian Mechanics
- [ ] **Equation**: `L = T - V` (kinetic - potential energy)
- [ ] **Application**: Energy-based dynamics formulation

### Hamiltonian
- [ ] **Equation**: `H = Σpᵢq̇ᵢ - L`
- [ ] **Application**: Energy-based optimal control

### Principle of Least Action
- [ ] **Equation**: `δS = δ∫L dt = 0`
- [ ] **Application**: Optimal trajectory derivation

---

## 🎲 Stochastic Processes

### Wiener Process (Brownian Motion)
- [ ] **Equation**: `dW ~ N(0,dt)`
- [ ] **Application**: Continuous random walk modeling

### Stochastic Differential Equation
- [ ] **Equation**: `dx = f(x,t)dt + g(x,t)dW`
- [ ] **Application**: Stochastic system dynamics

### Itô's Lemma
- [ ] **Equation**: `df = (∂f/∂t + μ∂f/∂x + ½σ²∂²f/∂x²)dt + σ∂f/∂x dW`
- [ ] **Application**: Stochastic chain rule

### Ornstein-Uhlenbeck Process
- [ ] **Equation**: `dx = θ(μ-x)dt + σdW`
- [ ] **Application**: Mean-reverting noise model

### Martingale Property
- [ ] **Equation**: `E[Xₜ₊₁|X₁,...,Xₜ] = Xₜ`
- [ ] **Application**: Unbiased value estimation

---

## 🌐 Differential Geometry (Advanced Robotics)

### Manifold Charts
- [ ] **Equation**: `φ: M → ℝⁿ`
- [ ] **Application**: Local coordinate systems

### Tangent Space
- [ ] **Equation**: `TₚM`
- [ ] **Application**: Velocity representations

### Lie Groups
- [ ] **Equation**: `SO(3)` rotation matrices, `SE(3)` rigid transforms
- [ ] **Application**: 3D rotations and poses

### Exponential Map
- [ ] **Equation**: `exp: 𝔰𝔬(3) → SO(3)`
- [ ] **Application**: Skew-symmetric to rotation matrix

### Geodesic
- [ ] **Equation**: `∇_γ̇γ̇ = 0`
- [ ] **Application**: Shortest path on manifold

### Riemannian Metric
- [ ] **Equation**: `ds² = gᵢⱼdxⁱdxʲ`
- [ ] **Application**: Distance on curved spaces

---

## 📚 Additional Resources

### Recommended Textbooks
- Linear Algebra: "Introduction to Linear Algebra" by Gilbert Strang
- Calculus: "Calculus" by James Stewart
- Probability: "Introduction to Probability" by Blitzstein & Hwang
- Optimization: "Convex Optimization" by Boyd & Vandenberghe
- Robotics: "Modern Robotics" by Lynch & Park
- RL: "Reinforcement Learning: An Introduction" by Sutton & Barto

### Online Resources
- Khan Academy (Foundations)
- 3Blue1Brown (Visual Intuition)
- MIT OpenCourseWare
- Stanford CS229, CS231n, CS234

--
