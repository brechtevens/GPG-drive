<!-- Driving simulator -->
## Driving simulator

### Prerequisites
The code is written using Python 3.9, and mainly uses the following Python libraries:
* CasADI
* Pyglet
* OpEn
* panocpy (optional, development build) 

You can install the GPGdrive package using

```
pip install .
```

Furthermore, panocpy can be installed though a provided .whl, using (for Linux)
```
pip install ./local_wheels/panocpy-0.0.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_27_x86_64.whl
```
and (for Windows)
```
pip install ./local_wheels/panocpy-0.0.2-cp39-cp39-win_amd64.whl 
```

### Running examples
To run an example script, perform
```
python examples/{name_example_script}.py
```

The simulator window will pop up. Here, the following hotkeys can be used:
* SPACE: pause and unpause the game
* P: take a screenshot
* B: toggle visualization of the bounding boxes of vehicles
* T: change trajectory visualization mode
* K: kill all processes
* ESCAPE: kill all processes and visualize the obtained data

The game always starts in the paused mode by default.

You can move your visualization window by dragging the mouse and zoom by scrolling. Finally, you can click specific vehicles to open a pop-up window, allowing to visualize additional vehicle-specific information such as their costs, constraints and state variables.

### Project structure

* **examples/:** main scripts for creating and running a traffic environment simulation

* **local_wheels/:** local wheels required for the driving simulator

* **screenshots/:** folder for all screenshots taken in the simulator

* **src/GPGdrive**: source code of the simulator, consisting of

  * **car.py** : Classes for representing different vehicle types, i.e. UserControlledCar and GNEPOptimizerCar
  * **collision.py** : Contains various collision avoidance constraints
  * **constraints.py** : General class for representing constraints
  * **dynamics.py** : Classes for representing the dynamics of a vehicle using a longitudinal model or a kinematic bicycle model
  * **experiment.py** : Classes for representing different experiments
  * **feature.py** : General class for representing cost function features
  * **gaussseidelsolver.py** : Class for the Gauss-Seidel solution methodology proposed in the thesis
  * **lagrangiansolver.py** : Class for an Augmented Lagrangian solution methodology using single optimization problem
  * **lane.py** : Class for representing a lane of a road
  * **learning.py** : Class for the online learning methodology
  * **logger.py** : Class for the logger
  * **penalty.py** : Class for the penalty handler
  * **road.py** : Class for representing a road consisting of multiple lanes
  * **trajectory.py** : Class for representing a trajectory of an object
  * **visualize.py** : Class for the main loop and the visualization of experiments
  * **visualize_data.py** : function for visualizing information of an experiment
  * **world.py** : Contains a class for representing traffic situations
  * **theme/:** buttons and checkboxes for the pyglet-gui package
  * **triangulation/:** the earcut-python package
  * **environments/:** folder for adding custom environments, such as a racetrack
  * **images/:** images used for visualization of the traffic environments
  * **helpers/:** helper scripts for visualizations
