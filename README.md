## Pendlum Simulation
![test](https://user-images.githubusercontent.com/34671120/93115823-616b5300-f6f7-11ea-8d10-cf92626dd734.png)

A simple simulation of the pendulum

# Requirements
* Scipy
* Numpy
* Matplotlib
* Python(3.8 later is better)

# Usage
```python singlependulum.py```
or
```python doublependulum_nlp.py```

# Function
- difeq
	- Differential equation of the pendulum(adjust the coefficient etc. if necessary)
- gen
	- Create a diagram from the coordinates of the two mass points of the pendulum
- animate
	- A function for `FuncAnimation` in matplotlib