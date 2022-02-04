<!-- Driving simulator -->
## Driving simulator

### Prerequisites
The code is written using Python 3.9, and mainly uses the following Python libraries:
* CasADI
* Pyglet
* panocpy or OpEn

The required packages can be installed using

```
pip install -r requirements.txt
```

**Note:** make sure to select the correct panocpy wheel in requirements.txt if you are using Windows instead of Linux.

### Project structure

* **environments/:** folder for adding custom environments, such as a racetrack

* **examples/:** main scripts for creating and running a traffic environment simulation

* **images/:** images used for visualization of the traffic environments

* **local_wheels/:** local wheels required for the driving simulator

* **screenshots/:** folder for all screenshots taken in the simulator

* **src/**: source code of the simulator, consisting of

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

  
