import casadi as cs
from . import feature as feature
from . import lane as lane
from . import constraints as constraints
import pyglet.gl as gl
import pyglet.graphics as graphics
import numpy as np
from pyglet import shapes, sprite
from .helpers.tree_generator import generate_trees, generate_trees_Gabor
from .helpers.visualize_helpers import centered_image


class Scene:
    def __init__(self):
        return

    def draw(self, magnify):
        return

class Highway(Scene):
    """
    A class used to represent a highway with multiple lanes

    Attributes
    ----------
    lanes : list
        the list of Lane objects of the highway
    n : list
        the normal vector on the highway direction
    """
    def __init__(self, p, q, w, nb_lanes, length_list=None):
        """
        Parameters
        ----------
        p : list
            the first point on the center line of the 'first' lane
        q : list
            the second point on the center line of the 'first' lane
        w : float
            the width of each lane
        nb_lanes : int
            the number of lanes of the highway
        length_list : list
            the lengths of the different lanes
        """
        center_lane = lane.StraightLane(p, q, w)
        self.lanes = [center_lane]
        for n in range(1, nb_lanes):
            self.lanes += [center_lane.shifted(n)]
        self.n = self.lanes[0].n
        if length_list is not None:
            assert(len(length_list) == nb_lanes)
            for i, L in enumerate(length_list):
                self.lanes[i].length = L
        self.setup_trees()

        
    def setup_trees(self):
        def is_forest_zone(position):
            distances = np.vstack(self.boundary_distances()(position))
            return np.min(distances) < -2.5
        self.tree_locations = generate_trees_Gabor(100, 200, 10, radius=4.5, valid_check=is_forest_zone)
        self.tree_batch = graphics.Batch()
        self.tree_sprites = []
        for tree_location in self.tree_locations:
            index = np.random.randint(1,5)
            self.tree_sprites.append(sprite.Sprite(centered_image('GPGdrive/images/Trees/Treedark{}.png'.format(index)), subpixel=True, batch=self.tree_batch))
            tree_size = np.random.uniform(3,6)
            self.tree_sprites[-1].scale = tree_size/256
            self.tree_sprites[-1].x, self.tree_sprites[-1].y = tree_location
            # sprite.rotation = -x[2]*180./math.pi
            # sprite.opacity = opacity

    def get_lanes(self):
        """ Return the number of lanes of the highway """
        return self.lanes

    def gaussian(self, width=0.5):
        """ Returns a gaussian cost feature penalizing deviations from the center line of the lane

        Parameters
        ----------
        width : float
            the width of the gaussian
        """
        @feature.feature
        def f(x, u, x_other, k):
            return self.lanes[0].gaussian(width)(x, u, x_other)
        return f

    def quadratic(self):
        """ Returns a quadratic cost feature penalizing deviations from the center line of the lane """
        @feature.feature
        def f(x, u, x_other, k):
            return self.lanes[0].quadratic()(x, u, x_other, k)
        return f

    def linear(self):
        """ Returns a linear cost feature penalizing driving along the normal vector """
        @feature.feature
        def f(x, u, x_other, k):
            return self.lanes[0].linear()(x, u, x_other)
        return f

    def boundary_distances(self):
        """ Returns the distance to the edges of the road

        Parameters
        ----------
        car : Car object
            the ego vehicle
        """
        @constraints.stageconstraints
        def h(x, u=None):
            edge1 = self.lanes[0].get_edges()[0]
            edge2 = self.lanes[-1].get_edges()[1]
            h = []
            h.append((x[0] - edge1[0]) * self.lanes[0].n[0] + (x[1] - edge1[1]) * self.lanes[0].n[1])
            h.append(- (x[0] - edge2[0]) * self.lanes[-1].n[0] - (x[1] - edge2[1]) * self.lanes[-1].n[1])
            return h
        h.length = 2
        h.type = "inequality"
        return h


    def boundary_constraint(self, car, *args):
        """ Returns the 8 inequality boundary constraints ensuring the car remains withing the boundaries of the highway

        Parameters
        ----------
        car : Car object
            the ego vehicle
        """
        @constraints.stageconstraints
        def h(x, u=None):
            edge1 = self.lanes[0].get_edges()[0]
            edge2 = self.lanes[-1].get_edges()[1]
            vehicle_corners = car.corners_x(x[car.id])
            h = []
            for corner in vehicle_corners:
                h.append((corner[0] - edge1[0]) * self.lanes[0].n[0] + (corner[1] - edge1[1]) * self.lanes[0].n[1])
                h.append(- (corner[0] - edge2[0]) * self.lanes[-1].n[0] - (corner[1] - edge2[1]) * self.lanes[-1].n[1])
            return h
        h.length = 8
        h.type = "inequality"
        h.id_list = (car.id,)
        return h

    def right_lane_constraint(self, car):
        """ Returns the equality constraint ensuring the car remains withing the boundaries of lane zero

        Parameters
        ----------
        car : Car object
            the ego vehicle
        """
        @constraints.stageconstraints
        def g(x, u=None):
            upper_edge = self.lanes[0].get_edges()[1]
            vehicle_corners = car.corners_x(x[car.id])
            n = self.lanes[0].n
            m = self.lanes[0].m
            g = []
            for corner in vehicle_corners:
                h1 = - (corner[0] - upper_edge[0]) * n[0] - (corner[1] - upper_edge[1]) * n[1]
                h2 = - (corner[0] - self.lanes[1].length) * m[0] - (corner[1] - upper_edge[1]) * m[1]
                g.append(cs.fmin(h1, 0) * cs.fmin(h2, 0))
                # g.append(cs.fmax(cs.fmin(h1, 0), cs.fmin(h2, 0)))
            return g
        g.length = 4
        g.type = "equality"
        g.id_list = (car.id,)
        return g

    def aligned(self, factor=1.):
        """ Returns a quadratic cost feature penalizing deviations from driving along the direction of the highway

        Parameters
        ----------
        factor : float
            the cost feature importance
        """
        @feature.feature
        def f(x, u, x_other, k):
            return - factor * (x[2] - cs.arctan2(-self.n[0], self.n[1]))**2
        return f

    def draw(self, magnify):
        """ Draws the road

        Parameters
        ----------
        magnify : float
            the manification of the visualizer
        """
        for lane in self.lanes:
            lane.draw_lane_surface()
        gl.glLineWidth(1/magnify)

        def left_line(lane):
            return np.hstack([lane.p - lane.m * lane.length + 0.5 * lane.w * lane.n,
                        lane.p + lane.m * lane.length + 0.5 * lane.w * lane.n])

        def right_line(lane):
            return np.hstack([lane.p - lane.m * lane.length - 0.5 * lane.w * lane.n,
                        lane.p + lane.m * lane.length - 0.5 * lane.w * lane.n])

        def end_line(lane):
            return np.hstack([lane.p + lane.m * lane.length - 0.5 * lane.w * lane.n,
                               lane.p + lane.m * lane.length + 0.5 * lane.w * lane.n])

        def draw_dashed_line(line, number_of_sections=1000):
            x0, y0, x1, y1 = line
            xs=np.linspace(x0,x1,number_of_sections+1)
            ys=np.linspace(y0,y1,number_of_sections+1)
            # batch = graphics.Batch()

            for i in range(number_of_sections//4):
                # shapes.Line(xs[4*i], ys[4*i], xs[4*i+1], ys[4*i+1], width = 1, batch=batch)
                graphics.draw(2, gl.GL_LINES, ('v2f', [xs[4*i], ys[4*i], xs[4*i+1], ys[4*i+1]]))

            # batch.draw()


        if len(self.lanes) == 1:
            self.lanes[0].draw_simple_lane_lines(1/magnify)
        else:
            for k, lane in enumerate(self.lanes):
                if k == 0:
                    gl.glColor3f(1., 1., 0.)
                    graphics.draw(2, gl.GL_LINES, ('v2f', right_line(lane)))
                    graphics.draw(2, gl.GL_LINES, ('v2f', end_line(lane)))
                elif k == len(self.lanes)-1:
                    gl.glColor3f(1., 1., 1.)
                    draw_dashed_line(right_line(lane))

                    #Hard coded parking slots for merging experiment
                    W = 43
                    for i in range(1):
                        graphics.draw(4, gl.GL_LINE_LOOP, ('v2f',
                           np.hstack(
                               [lane.p + lane.m * (
                                           W + 6 * i) - 0.5 * lane.w * lane.n,
                                lane.p + lane.m * (
                                            W + 6 * i) + 0.5 * lane.w * lane.n,
                                lane.p + lane.m * (
                                        W + 6 * (i + 1)) + 0.5 * lane.w * lane.n,
                                lane.p + lane.m * (
                                        W + 6 * (i + 1)) - 0.5 * lane.w * lane.n])
                           ))

                    gl.glColor3f(1., 1., 0.)
                    graphics.draw(2, gl.GL_LINES, ('v2f', left_line(lane)))
                    graphics.draw(2, gl.GL_LINES, ('v2f', end_line(lane)))
                else:
                    gl.glColor3f(1., 1., 1.)
                    draw_dashed_line(left_line(lane))
                    draw_dashed_line(right_line(lane))
                    gl.glColor3f(1., 1., 0.)
                    graphics.draw(2, gl.GL_LINES, ('v2f', end_line(lane)))
        gl.glColor3f(1., 1., 1.)

        self.tree_batch.draw()




class Crossroad(Scene):
    """
    A class used to represent a crossroad with multiple lanes

    Attributes
    ----------
    lanes : list
        the list of Lane objects of the crossroad
    n : list
        the normal vector on the 'main' crossroad direction
    """
    def __init__(self, p, q, w, nb_lanes):
        """
        Parameters
        ----------
        p : list
            the first point on the center line of the 'main' lane
        q : list
            the second point on the center line of the 'main' lane
        w : float
            the width of each lane
        nb_lanes : int
            the number of lanes on each side of the road
        """
        center_lane = lane.StraightLane(p, q, w)
        self.nb_lanes = nb_lanes
        self.lanes = [center_lane]
        for n in range(1, 2*nb_lanes):
            self.lanes.append(center_lane.shifted(n))
        self.n = self.lanes[0].n
        self.m = np.array([self.n[1], -self.n[0]])
        self.w = w

        center_lane_opposite = lane.StraightLane(p, p+self.n, w)
        self.lanes_opposite = [center_lane_opposite]
        for n in range(1, 2*nb_lanes):
            self.lanes_opposite.append(center_lane_opposite.shifted(n))

    def get_lanes(self):
        """ Return the lanes of the highway """
        return [*self.lanes, *self.lanes_opposite]

    def gaussian(self, width=0.5):
        """ Returns a gaussian cost feature penalizing deviations from the center line of the lane

        Parameters
        ----------
        width : float
            the width of the gaussian
        """
        @feature.feature
        def f(x, u, x_other, k):
            return self.lanes[0].gaussian(width)(x, u, x_other)
        return f

    def quadratic(self):
        """ Returns a quadratic cost feature penalizing deviations from the center line of the lane """
        @feature.feature
        def f(x, u, x_other, k):
            return self.lanes[0].quadratic()(x, u, x_other)
        return f

    def get_position_vehicle(self, lane_index, nb_cars, pos='l', velocity=0):
        positions = []
        if pos == 'l':
            delta_m = - 2 * self.nb_lanes
            delta_n = 0
            heading_direction = self.m
            lane_shift_direction = self.n
        elif pos == 'r':
            delta_m = 1
            delta_n = 2 * self.nb_lanes - 1
            heading_direction = -self.m
            lane_shift_direction = -self.n
        elif pos == 'u':
            delta_m = - 2 * self.nb_lanes + 1
            delta_n = 2 * self.nb_lanes
            heading_direction = -self.n
            lane_shift_direction = self.m
        elif pos == 'd':
            delta_m = 0
            delta_n = -1
            heading_direction = self.n
            lane_shift_direction = -self.m

        pos_center = self.lanes[0].p
        pos_at_start = pos_center + self.m * delta_m * self.w + self.n * delta_n * self.w
        pos_in_lane = pos_at_start + lane_index * lane_shift_direction * self.w
        for i in range(nb_cars):
            pos_car = pos_in_lane - 8 * i * heading_direction
            positions.append([pos_car[0], pos_car[1], np.arctan2(heading_direction[1], heading_direction[0]), velocity])
        return positions

    def boundary_constraint(self, car, check_all_corners=False):
        """ Returns the equality boundary constraints ensuring the car remains withing the boundaries of the crossroads

        Parameters
        ----------
        car : Car object
            the ego vehicle
        """
        @constraints.stageconstraints
        def g(x, u=None):
            edge1 = self.lanes[0].get_edges()[0]
            edge2 = self.lanes[-1].get_edges()[1]
            edge3 = self.lanes_opposite[0].get_edges()[0]
            edge4 = self.lanes_opposite[-1].get_edges()[1]
            g = []
            def eval_affine(point, lane, edge):
                return (point[0] - edge[0]) * lane.n[0] + (point[1] - edge[1]) * lane.n[1]

            if check_all_corners:
                vehicle_corners = car.corners_x(x[car.id])
                for corner in vehicle_corners:
                    g.append(cs.fmax(-eval_affine(corner, self.lanes[0], edge1), 0) * cs.fmax(-eval_affine(corner, self.lanes_opposite[0], edge3), 0))
                    g.append(cs.fmax(-eval_affine(corner, self.lanes[0], edge1), 0) * cs.fmax(eval_affine(corner, self.lanes_opposite[-1], edge4), 0))
                    g.append(cs.fmax(eval_affine(corner, self.lanes[-1], edge2), 0) * cs.fmax(-eval_affine(corner, self.lanes_opposite[0], edge3), 0))
                    g.append(cs.fmax(eval_affine(corner, self.lanes[-1], edge2), 0) * cs.fmax(eval_affine(corner, self.lanes_opposite[-1], edge4), 0))
            else:
                margin = 1
                point = x[car.id][0:2]
                g.append(cs.fmax(margin-eval_affine(point, self.lanes[0], edge1), 0) * cs.fmax(margin-eval_affine(point, self.lanes_opposite[0], edge3), 0))
                g.append(cs.fmax(margin-eval_affine(point, self.lanes[0], edge1), 0) * cs.fmax(margin+eval_affine(point, self.lanes_opposite[-1], edge4), 0))
                g.append(cs.fmax(margin+eval_affine(point, self.lanes[-1], edge2), 0) * cs.fmax(margin-eval_affine(point, self.lanes_opposite[0], edge3), 0))
                g.append(cs.fmax(margin+eval_affine(point, self.lanes[-1], edge2), 0) * cs.fmax(margin+eval_affine(point, self.lanes_opposite[-1], edge4), 0))
            return g
        g.length = 16 if check_all_corners else 4
        g.type = "equality"
        g.id_list = (car.id,)
        return g

    def draw(self, magnify):
        """ Draws the crossroad

        Parameters
        ----------
        magnify : float
            the manification of the visualizer
        """
        for lane in self.lanes:
            lane.draw_lane_surface()
        for lane in self.lanes_opposite:
            lane.draw_lane_surface()
        gl.glLineWidth(1/magnify)

        def left_line(lane):
            return np.hstack([lane.p - lane.m * lane.length + 0.5 * lane.w * lane.n,
                        lane.p + lane.m * lane.length + 0.5 * lane.w * lane.n])

        def right_line(lane):
            return np.hstack([lane.p - lane.m * lane.length - 0.5 * lane.w * lane.n,
                        lane.p + lane.m * lane.length - 0.5 * lane.w * lane.n])

        def end_line(lane):
            return np.hstack([lane.p + lane.m * lane.length - 0.5 * lane.w * lane.n,
                               lane.p + lane.m * lane.length + 0.5 * lane.w * lane.n])
        
        def draw_dashed_line(line, number_of_sections=1000):
            x0, y0, x1, y1 = line
            xs=np.linspace(x0,x1,number_of_sections+1)
            ys=np.linspace(y0,y1,number_of_sections+1)
            # batch = graphics.Batch()

            for i in range(number_of_sections//4):
                # shapes.Line(xs[4*i], ys[4*i], xs[4*i+1], ys[4*i+1], width = 1, batch=batch)
                graphics.draw(2, gl.GL_LINES, ('v2f', [xs[4*i], ys[4*i], xs[4*i+1], ys[4*i+1]]))

            # batch.draw()

        def split_line_intersection(line, opposing_lanes):
            def get_intersection(line1, line2):
                x1, y1, x2, y2 = line1
                x3, y3, x4, y4 = line2

                D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / D
                py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / D

                return [px, py]

            px1, py1 = get_intersection(line, right_line(opposing_lanes[0]))
            px2, py2 = get_intersection(line, left_line(opposing_lanes[-1]))
            
            x1, y1, x2, y2 = line
            if np.linalg.norm([x1 - px1, y1 - py1]) < np.linalg.norm([x1 - px2, y1 - py2]):
                return [np.array([x1, y1, px1, py1]), np.array([px2, py2, x2, y2])]
            else:
                return [np.array([x1, y1, px2, py2]), np.array([px1, py1, x2, y2])]

        if len(self.lanes) == 1:
            self.lanes[0].draw_simple_lane_lines(1/magnify)
        else:
            roads = [self.lanes, self.lanes_opposite]
            for j, lanes in enumerate(roads):
                for k, lane in enumerate(lanes):
                    if k == 0:
                        gl.glColor3f(1., 1., 0.)
                        for line in split_line_intersection(right_line(lane), roads[j-1]):
                            graphics.draw(2, gl.GL_LINES, ('v2f', line))
                        graphics.draw(2, gl.GL_LINES, ('v2f', end_line(lane)))
                    else:
                        if k != self.nb_lanes:
                            gl.glColor3f(1., 1., 1.)
                        for line in split_line_intersection(right_line(lane), roads[j-1]):
                            draw_dashed_line(line)
                        gl.glColor3f(1., 1., 0.)
                        graphics.draw(2, gl.GL_LINES, ('v2f', end_line(lane)))

                        if k == len(lanes)-1:
                            gl.glColor3f(1., 1., 0.)
                            for line in split_line_intersection(left_line(lane), roads[j-1]):
                                graphics.draw(2, gl.GL_LINES, ('v2f', line))
                            graphics.draw(2, gl.GL_LINES, ('v2f', end_line(lane)))
        gl.glColor3f(1., 1., 1.)
