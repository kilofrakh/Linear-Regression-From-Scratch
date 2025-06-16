import pandas as pd
import numpy as np
np.random.seed(0)
df = pd.DataFrame({
    'sutdytime': np.random.randint(1, 10, size=100),
    'score': np.random.randint(50, 100, size=100)
})


def gradientDescent(m_now, b_now, points, l):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].sutdytime
        y = points.iloc[i].score
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - l * m_gradient
    b = b_now - l * b_gradient
    return m, b

m = 0
b = 0
l = 0.01
epochs = 1000
for i in range(epochs):
    m,b = gradientDescent(m, b, df, l)
print(f"m: {m}, b: {b}")
