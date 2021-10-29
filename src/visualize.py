#!/usr/bin/env python
import pyglet
from pyglet.window import key
import pyglet.gl as gl
import pyglet.graphics as graphics
from pyglet import shapes
import numpy as np
import time
import math
import casadi as cs

import src.logger as logger
import src.collision as collision
import src.boundingbox as boundingbox
import src.car as car
import src.visualize_data as visualize_data
from src.helpers.visualize_helpers import centered_image

import matplotlib.cm
import os
import datetime
import multiprocessing as mp


from pyglet_gui.manager import Manager
from pyglet_gui.buttons import Button, OneTimeButton, Checkbox, GroupButton
from pyglet_gui.containers import VerticalContainer
from pyglet_gui.theme import Theme

working_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
pyglet.resource.path = [working_dir]
pyglet.resource.reindex()

# MULTIPROCESSING HACKS
_func = None


def worker_init(func):
    global _func
    _func = func


def worker(x):
    return _func(x)


class Visualizer(object):
    """
    A class used to visualize the behaviour of the vehicles over time in the traffic scenario
    """
    def __init__(self, experiment, fullscreen=False):
        """
        Parameters
        ----------
        experiment: Experiment object
            the experiment which should be handled
        fullscreen : boolean, optional
            determines whether the window should be fullscreen
        """
        # Set-up window settings
        self.width_factor = experiment.pyglet_visualization_settings.width_factor
        self.height_ratio = experiment.pyglet_visualization_settings.height_ratio
        self.width = int(1000 * self.width_factor)
        self.height = self.width // self.height_ratio
        self.window = pyglet.window.Window(self.width, self.height, fullscreen=fullscreen)
        # gl.glScalef(1.0, 1.0 * self.height_ratio, 0.0)
        self.window.on_draw = self.on_draw
        self.keys = key.KeyStateHandler()
        self.window.push_handlers(self.keys)
        self.window.on_key_press = self.on_key_press
        self.window.on_mouse_drag = self.on_mouse_drag
        self.window.on_mouse_scroll = self.on_mouse_scroll
        self.window.on_mouse_press = self.on_mouse_press
        self.magnify = experiment.pyglet_visualization_settings.magnify
        self.camera_center = None
        self.camera_offset = experiment.pyglet_visualization_settings.camera_offset

        self.togglables = {
            'show_live_data' : experiment.pyglet_visualization_settings.show_live_data,
            'show_bounding_box' : True,
            'show_aligned_box' : False,
            'show_avoidance_box' : False,
            'show_avoidance_ellipse' : False,
            'show_trajectory_mode' : 1,
            'heatmap_show' : False,
            'heatmap_show_constraints' : False,
            'show_heatmap' : None,
            'show_feasible_region' : None
        }
        self.live_data_border_size = experiment.pyglet_visualization_settings.live_data_border_size
        self.live_data_box_position = experiment.pyglet_visualization_settings.live_data_box_position

        # Set-up pyglet variables
        self.iters = 1000  # todo hard-coded
        self.grass = pyglet.resource.texture('images/grass.png')
        self.paused = False

        # Initialize variables for the lanes and for the cars along with their positions
        self.cars = [c for c in experiment.world.cars]
        self.scene = experiment.world.scene
        # self.roads = [r for r in experiment.world.roads]
        self.Ts = experiment.world.Ts
        self.visible_cars = []
        self.main_car = None

        # Initialize heatmap variables
        self.heatmap_x1 = None
        self.heatmap_x0 = None
        self.heat = None
        self.heatmap = None
        self.heatmap_valid = False
        self.cm = matplotlib.cm.jet
        self.heatmap_size = (256,256)

        # Settings for visualization windows when pressing 'ESC' button
        self.data_visualization_windows = experiment.data_visualization_windows

        # Setting for saving the experiment
        self.logger = logger.Logger(experiment)

        # Current iteration of the world
        self.current_iteration = 0

        # Define colors
        self.colors_dict = {'red': [1., 0., 0.], 'yellow': [1., 1., 0.], 'purple': [0., 0.5, 0.5],
                            'white': [1., 1., 1.], 'orange': [1., 0.5, 0.], 'gray': [0.2, 0.2, 0.2],
                            'blue': [0., 0.7, 1.]}
        self.batches = []

        # Initialize required variables for live data visualization
        self.live_data_shown = ['x', 'y', 'angle', 'velocity', 'acceleration', 'steering angle', 'stage cost', 'cost',
                                'effective constraint violation', 'planned constraint violation']
        [x_min, x_max, y_min, y_max] = self.initialize_live_data_window()
        self.text_box_background = pyglet.sprite.Sprite(pyglet.resource.image('images/gray_box.png'), 0, 0)
        self.text_box_background.scale_x = (x_max - x_min) / 150
        self.text_box_background.scale_y = (y_max - y_min) / 200
        self.text_box_background.position = (x_min, y_min)
        self.text_box_background.opacity = 150

        # Set main car and heatmap
        if experiment.pyglet_visualization_settings.id_main_car is not None:
            self.main_car = experiment.world.cars[experiment.pyglet_visualization_settings.id_main_car]

        # if isinstance(self.main_car, car.GPGOptimizerCar):
        #     self.set_heat(experiment.world.cars[experiment.pyglet_visualization_settings.id_main_car].reward)

        def car_sprite(color):
            """ Returns the sprite of a car

            Parameters
            ----------
            color : str
                the color of the car for the sprite
            """
            return pyglet.sprite.Sprite(centered_image('images/car-{}.png'.format(color)), subpixel=True)

        # Initialize sprites
        self.sprites = {c: car_sprite(c) for c in ['red', 'yellow', 'purple', 'white', 'orange', 'gray', 'blue']}

    def initialize_live_data_window(self):
        x_min = self.live_data_box_position[0]
        x_max = x_min + 2*self.live_data_border_size
        y_max = self.window.height - self.live_data_box_position[1]
        y_min = self.window.height - 2*self.live_data_border_size - 20*len(self.live_data_shown) - 20

        def make_label(obj, name, nb_shifted):
            setattr(obj, 'label_' + name,
                    pyglet.text.Label(
                        name + ':  N/A',
                        font_name='Georgia',
                        font_size=12,
                        x=x_min + self.live_data_border_size,
                        y=y_max - self.live_data_border_size - 20*nb_shifted,
                        anchor_x='left', anchor_y='top',
                        color=(0, 0, 0, 255)
                    ))
        for shift, item in enumerate(self.live_data_shown):
            make_label(self, item, shift)
            x_max = max(x_max, x_min + 2*self.live_data_border_size + getattr(self, 'label_' + item).content_width)
        return [x_min, x_max, y_min, y_max]

    def kill_all(self):
        """ Kills the TCP servers of the OpEn optimizers of the GPGOptimizerCars, if any """
        for car in reversed(self.cars):
            try:
                for i in car.optimizer.id_list:
                    car.optimizer.solver_dict[i].kill()
            except:
                pass
            try:
                for i in car.optimizer.id_list:
                    car.observer.observation_solver_dict[i].kill()
            except:
                pass

    def save_screenshot(self, folder_name='screenshots', index=None):
        """ Saves a screenshot

        Parameters
        ----------
        folder_name : str
            the location of the folder to save the screenshot
        index : int
            the index of the image
        """
        if index is None:
            time_info = datetime.datetime.now()
            index = time_info.month*1e8 + time_info.day*1e6 + time_info.hour*1e4 + time_info.minute*1e2 +\
                time_info.second
        # Capture image from pyglet
        pyglet.image.get_buffer_manager().get_color_buffer().save(folder_name + '/screenshot-%d.png' % index)

    def on_key_press(self, symbol, *args):
        """ Defines what should be performed when a key is pressed

        Parameters
        ----------
        symbol : key attribute
            the pressed key
        """
        # the ESC button stops the experiment, kills all TCP servers and visualizes the requested information
        if symbol == key.ESCAPE:
            self.window.close()
            self.kill_all()
            pyglet.app.exit()
            if self.data_visualization_windows is not None:
                visualize_data.plot(self.logger.history, self.data_visualization_windows)

        # the P button saves a screenshot
        if symbol == key.P:
            self.save_screenshot()

        # the SPACE button pauses and unpauses the game
        if symbol == key.SPACE:
            self.paused = not self.paused

        # the H button can be used to show a heatmap
        # if symbol == key.H:
        #     self.togglables['heatmap_show'] = not self.togglables['heatmap_show']
        #     if self.togglables['heatmap_show']:
        #         self.heatmap_valid = False

        # the T button can be used to change the visualization of the trajectories
        if symbol == key.T:
            self.togglables['show_trajectory_mode'] = (self.togglables['show_trajectory_mode']+1)%4

        # the B button can be used to show bounding boxes of vehicles
        if symbol == key.B:
            self.togglables['show_bounding_box'] = not self.togglables['show_bounding_box']

        # the K button can be used to kill all processes and close the window
        if symbol == key.K:
            self.window.close()
            self.kill_all()
            pyglet.app.exit()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.camera_center is None:
            self.camera_center = self.center()
        self.camera_center[0] -= (dx/self.width)*80*self.magnify
        self.camera_center[1] -= (dy/self.height)*(80/self.height_ratio)*self.magnify
        self.heatmap_valid = False

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.magnify *= (1 - 0.1*scroll_y)
        self.heatmap_valid = False

    def on_mouse_press(self, x, y, scroll_x, scroll_y):
        o = self.convert_pixel_to_coordinate(x, y)
        for car in self.cars:
            equality_constraint = collision.product_formulation_equality_constraint_of_target_vehicle(0)(car, car.x, o)
            if equality_constraint != 0:
                if hasattr(car, "manager"):
                    car.manager.delete()
                    delattr(car, "manager")
                else:
                    car.manager = self.open_car_manager(car)

    def control_loop(self, _=None):
        if self.paused:
            return
        if self.iters is not None and self.current_iteration >= self.iters:
            pyglet.app.exit()
            return
        print('__________NEW GAME ITERATION__________')
        steer = 0.
        gas = 0.

        # Control the UserControlledCars using the arrow keys
        if self.keys[key.UP]:
            gas += 1.
        if self.keys[key.DOWN]:
            gas -= 1.
        if self.keys[key.LEFT]:
            steer += 0.2
        if self.keys[key.RIGHT]:
            steer -= 0.2

        # Set heatmap false again
        self.heatmap_valid = False

        # Calculate control actions for each vehicle
        for vehicle in reversed(self.cars):
            vehicle.control(steer, gas)

        # Update iteration
        self.current_iteration += 1

        # Log the data from the cars
        self.logger.log_data(self.cars)
        exit_status = self.logger.write_data_to_files(self.current_iteration)

        # Let vehicles observe actions of other drivers
        for vehicle in self.cars:
            if isinstance(vehicle, car.GPGOptimizerCar):
                vehicle.observe()

        # Move cars
        for vehicle in self.cars:
            vehicle.move()

        # Update cars
        for vehicle in self.cars:
            vehicle.update(self.current_iteration) 

        # Stop experiment when logger has completed
        if exit_status == 1:
            pyglet.app.exit()

    def center(self):
        """ Returns the 'center' for the camera """
        if self.camera_center is not None:
            return self.camera_center[0:2]
        elif self.main_car is None:
            return cs.DM(self.camera_offset)
        else:
            return cs.vertcat(self.main_car.center[0] + self.camera_offset[0], self.camera_offset[1])

    def camera_vision_vertices(self):
        o = self.center()
        return [o[0]-40*self.magnify, o[0]+40*self.magnify,
                o[1]-40*self.magnify/self.height_ratio, o[1]+40*self.magnify/self.height_ratio]

    def convert_pixel_to_coordinate(self, x, y):
        o = self.center()
        return cs.vertcat(o[0] + (x/self.width - 0.5) * 80*self.magnify, o[1] + (y/self.height - 0.5) * 80*self.magnify/self.height_ratio)

    def convert_coordinate_to_pixel(self, x, y):
        o = self.center()
        return cs.vertcat(self.width * (0.5 + (1/(80*self.magnify)) * (x - o[0])), self.height * (0.5 + (self.height_ratio/(80*self.magnify)) * (y - o[1])))

    def camera(self):
        """ Sets camera """
        gl.glOrtho(*self.camera_vision_vertices(), -1., 1.)

    def set_heat(self, f, id):
        """ Sets heatmap function """
        def val(p, x_dict):
            x_dict[id] = [p[0], p[1], self.cars[id].x[2], self.cars[id].x[3]]
            return f([p[0], p[1], 0, 0], [0, 0], x_dict, 0)
        self.heat = val

    def set_heat_constraint(self, constraints, id):
        """ Sets heatmap function """
        def val(p, x_dict):
            x_dict[id] = [p[0], p[1], self.cars[id].x[2], self.cars[id].x[3]]
            return constraints(x_dict, [0, 0])
        self.heat = val

    def draw_heatmap(self, vehicle, show_constraints=False):
        """ Draws the heatmap """
        if self.heatmap_valid and (vehicle.id != self.heatmap_id or self.togglables['heatmap_show_constraints'] != show_constraints):
            self.heatmap_valid = False
        if not self.heatmap_valid:
            if show_constraints:
                self.set_heat_constraint(vehicle.get_current_constraint_violation, vehicle.id)
            else:
                self.set_heat(vehicle.get_current_reward, vehicle.id)
            self.heatmap_id = vehicle.id
            self.togglables['heatmap_show_constraints'] = show_constraints
            o = self.center()
            x0 = o - np.asarray([40., 40. / self.height_ratio], dtype=object) * self.magnify
            x1 = o + np.asarray([40., 40. / self.height_ratio], dtype=object) * self.magnify
            self.heatmap_x0 = x0
            self.heatmap_x1 = x1
            values = self.compute_heatmap_values(vehicle, x0, x1)
            self.heatmap = self.values_to_img(values).get_texture()
            self.heatmap_valid = True
        gl.glEnable(self.heatmap.target)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glBindTexture(self.heatmap.target, self.heatmap.id)
        gl.glColor4f(1.0, 1.0, 1.0, 0.9)
        graphics.draw(4, gl.GL_QUADS,
                      ('v2f', (self.heatmap_x0[0], self.heatmap_x0[1], self.heatmap_x1[0], self.heatmap_x0[1],
                               self.heatmap_x1[0], self.heatmap_x1[1], self.heatmap_x0[0], self.heatmap_x1[1])),
                      ('t2f', (0., 0., 1., 0., 1., 1., 0., 1.)), )
        gl.glDisable(self.heatmap.target)

    def compute_heatmap_values(self, vehicle, x0, x1):
        x_range = np.linspace(x0[0], x1[0], self.heatmap_size[0])
        y_range = np.linspace(x0[1], x1[1], self.heatmap_size[1])

        # x_grid, y_grid = np.meshgrid(x_range, y_range)
        # positions = np.vstack((x_grid.ravel(), y_grid.ravel()))

        values = np.zeros(self.heatmap_size)
        x_dict = {}
        for i in vehicle.players:
            x_dict[i] = self.cars[i].x
        for i in vehicle.obstacles:
            x_dict[i] = self.cars[i].x

        func = lambda pt: self.heat(pt, x_dict)

        #with mp.Pool(None, initializer=worker_init, initargs=(func,)) as p:
        #worker_init(func)
        #with mp.Pool() as p:
        #    values = p.map(worker, positions.T)
        #values = np.reshape(values, self.heatmap_size)

        for i, x in enumerate(np.linspace(x0[0], x1[0], self.heatmap_size[0])):
            for j, y in enumerate(np.linspace(x0[1], x1[1], self.heatmap_size[1])):
                values[j, i] += 0 if func([x, y]).is_empty() else func([x, y])
                # values[j, i] += 0 if func([x, y]).is_empty() else (0 if func([x, y]) <= 0 else func([x, y]))
                # values[j, i] += 0 if func([x, y]).is_empty() else (0 if func([x, y]) <= 0 else 1)
        return values

    def values_to_img(self, values):
        values = (values-np.min(values))/(np.max(values)-np.min(values)+1e-6)
        values = self.cm(values)
        values[:, :, 3] = 0.7
        values = (values*255).astype(np.uint8)
        img = pyglet.image.ImageData(self.heatmap_size[0], self.heatmap_size[1], 'RGBA', values.tobytes())
        return img

    def draw_car(self, x, car, color='yellow', opacity=255):
        """ Draws a car with the given color at the given position

        Parameters
        ----------
        x : CasADi MX
            the given state vector for the vehicle [x, y, angle, ...]
        color : str, optional
            the color of the sprite
        opacity : int, optional
            the opacity of the sprite
        """
        sprite = self.sprites[color]
        sprite.scale = car.len/600
        sprite.x, sprite.y = x[0], x[1]
        sprite.rotation = -x[2]*180./math.pi
        sprite.opacity = opacity
        sprite.draw()

    def draw_car_id(self, x, car, color='yellow', opacity=255):
        """ Draws a car with the given color at the given position

        Parameters
        ----------
        x : CasADi MX
            the given state vector for the vehicle [x, y, angle, ...]
        color : str, optional
            the color of the sprite
        opacity : int, optional
            the opacity of the sprite
        """
        px = self.convert_coordinate_to_pixel(x[0], x[1])
        pyglet.text.Label(
            str(car.id),
            font_name='Georgia',
            font_size= 2*car.len / self.magnify,
            x=px[0],
            y=px[1],
            anchor_x="center", anchor_y="center",
            color=(0, 0, 0, opacity)
        ).draw()                  

    def draw_bounding_box(self, vertices, color='yellow'):
        """ Draws the rectangular bounding box

        Parameters
        ----------
        vertices : list
            the vertices of the bounding box
        color : str, optional
            the color of the bounding box
        """
        gl.glColor3f(self.colors_dict[color][0], self.colors_dict[color][1], self.colors_dict[color][2])
        gl.glBegin(gl.GL_LINE_LOOP)
        for vertex in vertices:
            gl.glVertex2f(vertex[0], vertex[1])
        gl.glEnd()
        gl.glColor3f(1., 1., 1.)

    def draw_ellipsoid(self, cx, cy, rx, ry, color):
        gl.glColor3f(self.colors_dict[color][0], self.colors_dict[color][1], self.colors_dict[color][2])

        num_segments = 25
        theta = 2*np.pi / num_segments
        c = np.cos(theta)
        s = np.sin(theta)

        x = 1
        y = 0

        gl.glBegin(gl.GL_LINE_LOOP)
        for i in range(num_segments):
            gl.glVertex2f(x * rx + cx, y * ry + cy)

            #apply the rotation matrix
            t = x
            x = c * x - s * y
            y = s * t + c * y
        gl.glEnd()
        gl.glColor3f(1., 1., 1.)

    def draw_trajectory(self, car, color):
        if self.togglables['show_trajectory_mode'] == 0:
            return
        if self.togglables['show_trajectory_mode'] == 1:
            self.draw_trajectory_line(car, color)
        if self.togglables['show_trajectory_mode'] == 2:
            self.draw_trajectory_faded(car, color)
        if self.togglables['show_trajectory_mode'] == 3:
            self.draw_past_trajectory_faded(car, color)

    def draw_trajectory_line(self, car, color):
        """ Draws the given trajectory

        Parameters
        ----------
        car : Car object
            the given car
        color : str, optional
            the color of the trajectory
        """
        if car.N > 1:
            trajectory = car.get_future_trajectory()
            if (isinstance(trajectory[0], cs.SX) or isinstance(trajectory[0], cs.MX)):
                if car.id in car.p_dict:
                    trajectory = [cs.DM(cs.substitute(state, car.p_dict[car.id], car.p_numeric_dict[car.id])).toarray(True) for state in trajectory]
            gl.glColor3f(self.colors_dict[color][0], self.colors_dict[color][1], self.colors_dict[color][2])

            # Draw a line strip
            gl.glLineStipple(5, 0x5555)
            gl.glEnable(gl.GL_LINE_STIPPLE)
            gl.glBegin(gl.GL_LINE_STRIP)
            gl.glVertex2f(car.x[0], car.x[1])
            for state in trajectory:
                gl.glVertex2f(state[0], state[1])
            gl.glEnd()
            gl.glDisable(gl.GL_LINE_STIPPLE)

            # Draw nodes at each sampling time
            gl.glEnable(gl.GL_POINT_SMOOTH)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glBegin(gl.GL_POINTS)
            # gl.glVertex2f(car.x[0].toarray()[0,0], car.x[1].toarray()[0,0])
            gl.glVertex2f(car.x[0], car.x[1])
            for state in trajectory:
                gl.glVertex2f(state[0], state[1])
            gl.glEnd()
            gl.glColor3f(1., 1., 1.)

    def draw_trajectory_faded(self, car, color):
        """ Draws the given trajectory

        Parameters
        ----------
        car : Car object
            the given car
        color : str, optional
            the color of the trajectory
        """
        if car.N > 1:
            trajectory = car.get_future_trajectory()
            if isinstance(trajectory[0], cs.SX) or isinstance(trajectory[0], cs.MX) and car.id in car.p_dict:
                trajectory = [cs.DM(cs.substitute(state, car.p_dict[car.id], car.p_numeric_dict[car.id])).toarray(True) for state in trajectory]

            sprite = self.sprites[color]
            opacity_list = np.linspace(100,0,len(trajectory)+2, dtype = int)[1:-1][::-1]

            for index, state in enumerate(trajectory[::-1]):
                center_x = car.center_x(state)
                sprite.x, sprite.y = center_x[0], center_x[1]
                sprite.rotation = -center_x[2] * 180. / math.pi
                sprite.opacity = opacity_list[index]
                sprite.draw()

    def draw_past_trajectory_faded(self, car, color):
        """ Draws the given trajectory

        Parameters
        ----------
        car : Car object
            the given car
        color : str, optional
            the color of the trajectory
        """
        faded_factor = 2
        trajectory = [[self.logger.history['x'][car.id][faded_factor*(-1-i)],
                       self.logger.history['y'][car.id][faded_factor*(-1-i)],
                       self.logger.history['angle'][car.id][faded_factor*(-1-i)],
                       self.logger.history['velocity'][car.id][faded_factor*(-1-i)]]
                      for i in range(min(self.current_iteration//2, 5))]
        sprite = self.sprites[color]
        opacity_list = np.linspace(150, 0, len(trajectory) + 1, dtype=int)[:-1][::-1]

        for index, state in enumerate(trajectory[::-1]):
            center_x = car.center_x(state)
            sprite.x, sprite.y = center_x[0], center_x[1]
            sprite.rotation = -center_x[2] * 180. / math.pi
            sprite.opacity = opacity_list[index]
            sprite.draw()

    def on_draw(self):
        """ Draws all objects and the background on the window """
        self.window.clear()
        gl.glColor3f(1., 1., 1.)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        self.camera()

        # Draw grass
        gl.glEnable(self.grass.target)
        gl.glEnable(gl.GL_BLEND)
        gl.glBindTexture(self.grass.target, self.grass.id)
        [x_min, x_max, y_min, y_max] = self.camera_vision_vertices()
        x_repeats = math.ceil(self.width/128.)
        y_repeats = math.ceil(self.height/128.)
        graphics.draw(4, gl.GL_QUADS,
            ('v2f', (x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max)),
                      ('t2f', (0., 0., x_repeats, 0., x_repeats, y_repeats, 0., y_repeats)),)
        gl.glDisable(self.grass.target)

        # Draw scene
        self.scene.draw(self.magnify)

        # Draw heatmap
        if self.togglables['show_heatmap'] is not None:
            self.draw_heatmap(self.cars[self.togglables['show_heatmap']], False)

        if self.togglables['show_feasible_region'] is not None:
            self.draw_heatmap(self.cars[self.togglables['show_feasible_region']], True)

        # for vehicle in self.cars:
        #     if hasattr(vehicle, "manager"):
        #         if vehicle.manager.controllers[0].is_pressed:
        #             self.draw_heatmap(vehicle, False)
        #         elif vehicle.manager.controllers[1].is_pressed:
        #             self.draw_heatmap(vehicle, True)

        # Draw cars
        for vehicle in self.cars:
            gl.glLineWidth(0.25 * vehicle.len * self.width_factor / self.magnify)
            gl.glPointSize(vehicle.len * self.width_factor / self.magnify)
            self.draw_trajectory(vehicle, vehicle.color)

        for vehicle in self.cars:
            self.draw_car(vehicle.center, vehicle, vehicle.color)
            gl.glLineWidth(0.25 * vehicle.len * self.width_factor / self.magnify)
            if self.togglables['show_bounding_box']:
                self.draw_bounding_box(vehicle.corners, vehicle.color)
            if self.togglables['show_aligned_box']:
                self.draw_bounding_box(boundingbox.get_axis_aligned_box_vertices(vehicle), vehicle.color)
        
            for vehicle2 in self.cars:
                if vehicle != vehicle2:
                    if self.togglables['show_avoidance_box']:
                        self.draw_bounding_box(boundingbox.get_axis_aligned_bounding_box_vertices(vehicle, vehicle2), vehicle.color)
                    if self.togglables['show_avoidance_ellipse']:
                        center_x, center_y, lambda_1, lambda_2 = boundingbox.get_axis_alligned_ellipsoid_vertices(vehicle, vehicle2)
                        self.draw_ellipsoid(center_x, center_y, np.sqrt(1/lambda_1), np.sqrt(1/lambda_2), vehicle.color)

        gl.glPopMatrix()

        for vehicle in self.cars:
            self.draw_car_id(vehicle.center, vehicle, vehicle.color)

        # Draw extra information about Speed and Headway distance
        self.draw_live_data_window()

        # Save image if save_on_draw
        if self.logger.save_on_draw:
            video_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'experiments', self.logger.settings.name_experiment, "video")
            if not self.logger.generate_video:
                self.save_screenshot(folder_name=video_path, index=self.current_iteration)
                self.logger.save_on_draw = False
            else:
                # Generate and save video
                os.system("ffmpeg -r " + str(1/self.Ts) + " -i " + video_path + "/screenshot-%01d.png -c:v libx264 -vf fps=25 -s " + str(self.width) + "x" + str(self.height) + " -pix_fmt yuv420p -crf 5 " + video_path + "/video.mp4") # For Windows, use e.g. "C:/ffmpeg/bin/ffmpeg.exe" instead of "ffmpeg"
                time.sleep(0.1)
                self.logger.generate_video = False
        
        for batch in self.batches:
            batch.draw()

    def draw_live_data_window(self):
        if self.togglables['show_live_data'] is not None:
            self.text_box_background.draw()
            for item in self.live_data_shown:
                try:
                    setattr(getattr(self, 'label_' + item), 'text', item + ': %.2f'%self.logger.history[item][self.togglables['show_live_data']][-1])
                except:
                    setattr(getattr(self, 'label_' + item), 'text', item + ': N/A')
                getattr(self, 'label_' + item).draw()

    def reset(self):
        """ Resets the variables of the visualizer """
        self.paused = True
        self.current_iteration = 0
        self.logger.reset(self.cars)
        for car in self.cars:
            car.reset()

    def run(self):
        """ Resets the visualized and runs the event loop """
        self.reset()
        pyglet.clock.schedule_interval(self.control_loop, self.Ts)
        # pyglet.clock.schedule(self.control_loop)

        pyglet.app.run()

    def open_car_manager(self, car):
        self.batches.append(pyglet.graphics.Batch())
        theme = Theme({"font": "Lucida Grande",
                    "font_size": 12,
                    "text_color": [255, 255, 255, 255],
                    "gui_color": [*[int(255*rgb) for rgb in self.colors_dict[car.color]],255],
                    "button": {
                        "down": {
                            "image": {
                                "source": "button-down.png",
                                "frame": [6, 6, 3, 3],
                                "padding": [12, 12, 4, 2]
                            },
                            "text_color": [0, 0, 0, 255]
                        },
                        "up": {
                            "image": {
                                "source": "button.png",
                                "frame": [6, 6, 3, 3],
                                "padding": [12, 12, 4, 2]
                            }
                        }
                    },
                    "checkbox": {
                        "checked": {
                            "image": {
                                "source": "checkbox-checked.png"
                            }
                        },
                        "unchecked": {
                            "image": {
                                "source": "checkbox.png"
                            }
                        }
                    }
                    }, resources_path=pyglet.resource.path[0] + '/theme/')

        # Set up a Manager
        # manager = Manager(VerticalContainer([Button(label="Persistent button"),
        #                         OneTimeButton(label="One time button"),
        #                         Checkbox(label="Checkbox"),
        #                         GroupButton(group_id='1', label="Group 1:Button 1"),
        #                         GroupButton(group_id='1', label="Group 1:Button 2"),
        #                         GroupButton(group_id='2', label="Group 2:Button 1"),
        #                         GroupButton(group_id='2', label="Group 2:Button 2"),
        #                         GroupButton(group_id='2', label="Group 2:Button 3"),
        #                         ]),
        #         window=self.window,
        #         batch=self.batches[-1],
        #         theme=theme)

        def on_press_boolean(togglable_name):
            def on_press(is_pressed):
                self.togglables[togglable_name] = is_pressed
            return on_press

        def on_press_id(togglable_name):
            def on_press(is_pressed):
                if is_pressed:
                    self.togglables[togglable_name] = car.id
                else:
                    self.togglables[togglable_name] = None
            return on_press

        manager = Manager(VerticalContainer([GroupButton(group_id='Heatmap'+str(car.id), label="Show heatmap", on_press = on_press_id('show_heatmap')),
                                             GroupButton(group_id='Heatmap'+str(car.id), label="Show feasible region", on_press = on_press_id('show_feasible_region')),
                                             Button(label="Show bounding box", is_pressed = self.togglables['show_bounding_box'], on_press = on_press_boolean('show_bounding_box')),
                                             Button(label="Show aligned bounding box", is_pressed = self.togglables['show_aligned_box'], on_press = on_press_boolean('show_aligned_box')),
                                             Button(label="Show avoidance box", is_pressed = self.togglables['show_avoidance_box'], on_press = on_press_boolean('show_avoidance_box')),
                                             Button(label="Show avoidance ellipse", is_pressed = self.togglables['show_avoidance_ellipse'], on_press = on_press_boolean('show_avoidance_ellipse')),
                                             Button(label="Show live data", is_pressed = self.togglables['show_live_data'], on_press = on_press_id('show_live_data'))]),
                window=self.window,
                batch=self.batches[-1],
                theme=theme)
        return manager
