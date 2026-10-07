# 02 - Machine learning (course Modules 5-12)

Hands-on: Labs [01](../../part1-ai-foundations/lab01_logistic_regression_iris.py) and [02](../../part1-ai-foundations/lab02_iris_pipeline_scaling_eval.py).

## How ML works
Features (input) + label (output) -> **training dataset** -> model learns the relationship -> **trained model** -> **inference** on unseen data.

| Type | Data | Typical uses |
|---|---|---|
| **Supervised** | labeled | disease detection, weather/stock forecasting, spam detection, credit scoring |
| **Unsupervised** | unlabeled | customer segmentation, outlier/fraud detection, targeted marketing |
| **Reinforcement** | rewards/penalties | robots, autonomous driving, game playing |

Traditional rules engine vs ML: rules are slow to build/update and need experts (but transparent); ML learns from historical decisions.

## Supervised - linear regression (continuous output)
- Example: house size -> price. One row = one training example; the table = training dataset.
- Line: **f(x) = w*x + b** - `w` slope/weight, `b` bias/intercept. Tilt with `w`, shift with `b`.
- **Error** = actual - predicted. **Loss** = penalty for a bad prediction (0 if perfect). Common loss = squared difference. Training iteratively adjusts `w`, `b` to minimise it.
- Continuous output -> **regression**; categorical output -> **classification**.

## Supervised - classification and logistic regression
- Binary (2 classes, spam/not spam) vs multi-class (>2, sentiment).
- Logistic regression outputs a probability through the **sigmoid** (S-curve squashing any number into 0-1). Threshold (usually 0.5): 6 h study -> 80% -> Pass; 4 h -> 20% -> Fail.
- **Iris**: 150 rows, 3 species (setosa, versicolor, virginica), 4 attributes (sepal/petal length and width) -> multi-class problem.

## Tooling (Modules 8-10)
- **Anaconda**: package management, isolated environments, Navigator GUI, cross-platform. **Jupyter**: live code + equations + visuals + narrative.
- Workflow: *load -> preprocess -> train -> evaluate -> predict*. Libraries: `pandas`, `scikit-learn`, `numpy`.
- **Standardization** (mean 0, std 1) stops large-magnitude features (square footage 1,000-5,000) dominating small ones (bedrooms 1-6).
- **train_test_split** with `random_state` = reproducible split. **accuracy_score** = correct / total on unseen data; detects **overfitting** (good on training, bad on new data). Demo result: accuracy 1.0.

## Unsupervised learning (Module 11)
- Finds structure without labels. **Clustering** groups similar items; unmatched items are **outliers**.
- Use cases: market segmentation, outlier analysis (fraud), recommendation systems.
- **Similarity** 0-1 (closer to 1 = more alike). Metrics: Euclidean, Manhattan, cosine, Jaccard.
- Workflow: (1) prepare data (missing values, normalise, scale) (2) build similarity matrix (3) run clustering - partition-, hierarchical-, density-, or distribution-based (4) interpret and iterate (no ground truth, so it is exploratory).

## Reinforcement learning (Module 12)
**Agent** (learner) - **Environment** - **State** - **Actions** - **Policy** (the agent's "brain": state -> action). Goal: the **optimal policy** that maximises reward, learned with Q-learning / Deep Q-learning. Applications: autonomous vehicles, smart assistants, industrial automation, adaptive game AI.
Robotic-arm example: set environment -> define state -> define action space -> define rewards/penalties -> train from random actions toward high-reward ones.
