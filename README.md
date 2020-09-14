## Pendlum Simulation

A simple simulation of the pendulum
The contents are

- singlependulum.py
Simulation of singlependulum 
- doublependulum_nlp.py
Simulation of doublependulum

# Requirements
* Scipy
* Numpy
* Matplotlib
* Python(3.8 later is better)

# Usage
```python singlependulum.py```
or
```python doublependulum.py```

# Function
- difeq
	- Differential equation of the pendulum(adjust the coefficient etc. if necessary)
- gen
	- Create a diagram from the coordinates of the two mass points of the pendulum
- animate
	- A function for `FuncAnimation` in matplotlib