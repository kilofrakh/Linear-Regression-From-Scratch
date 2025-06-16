import pandas as pd
import numpy as np
df = pd.read_csv('framingham (1).csv')

def lossfunction(m, b, points):
    totalError =0
    for i in range(len(points)):
        x = points.iloc[i, 0].sutdytime
        y = points.iloc[i].score
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def gradientDescent(points, m, b, learningRate):
    m_gradient = 0
    b_gradient = 0
    N = float(len(points))
    
    for i in range(len(points)):
        x = points.iloc[i, 0].sutdytime
        y = points.iloc[i].score
        m_gradient += -(2/N) * x * (y - (m * x + b))
        b_gradient += -(2/N) * (y - (m * x + b))
    
    m -= learningRate * m_gradient
    b -= learningRate * b_gradient
    
    return m, b

def run(points, starting_m, starting_b, learningRate, num_iterations):
    m = starting_m
    b = starting_b
    
    for i in range(num_iterations):
        m, b = gradientDescent(points, m, b, learningRate)
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {lossfunction(m, b, points)}")
    
    return m, b

def main():
    points = df[['sutdytime', 'score']]
    points.columns = ['sutdytime', 'score']
    
    starting_m = 0.0
    starting_b = 0.0
    learningRate = 0.01
    num_iterations = 1000
    
    m, b = run(points, starting_m, starting_b, learningRate, num_iterations)
    
    print(f"Final parameters: m = {m}, b = {b}")
if __name__ == "__main__":
    main()
    