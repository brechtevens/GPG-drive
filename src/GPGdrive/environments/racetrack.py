from GPGdrive.dataloader import loadMatrix
import scipy.io
import numpy as np
import os

from OpenGL.arrays import vbo
import pyglet.gl as gl
import pyglet.graphics as graphics
import GPGdrive.triangulation.earcut.earcut as earcut
import GPGdrive.feature as feature
import casadi as cs


def load_reference_track(track_name):
    # # Load in reference track + local curvature (expressed as an angle):
    # c: center, s: shortest, u: curvature, o: optimal

    # if track_name == 'c':
    #     dataLoad = loadMatrix('environments/racetrack/center_reference.txt', ' ', 'str')
    #     meta = scipy.io.loadmat('environments/racetrack/center_reference_Nseg.mat')
    #     simulation_steps = meta['Nseg'][0][0]-1
    # elif track_name == 's':
    #     dataLoad = loadMatrix('environments/racetrack/shortest_reference.txt', ' ', 'str')
    #     meta = scipy.io.loadmat('environments/racetrack/shortest_reference_Nseg.mat')
    #     simulation_steps = meta['Nseg'][0][0]-1
    # elif track_name == 'u':
    #     dataLoad = loadMatrix('environments/racetrack/curvature_reference.txt', ' ', 'str')
    #     meta = scipy.io.loadmat('environments/racetrack/curvature_reference_Nseg.mat')
    #     simulation_steps = meta['Nseg'][0][0]-1
    # else:
    #     dataLoad = loadMatrix('environments/racetrack/optimal_reference.txt', ' ', 'str')
    #     meta = scipy.io.loadmat('environments/racetrack/optimal_reference_Nseg.mat')
    #     simulation_steps = meta['Nseg'][0][0]-1

    working_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    centerline = scipy.io.loadmat(working_dir + '/environments/racetrack/centerline.mat')
    N = centerline['Nseg'][0,0]
    # centerX = dataLoad[0][:][:simulation_steps]
    # centerY = dataLoad[1][:][:simulation_steps]
    # theta = dataLoad[2][:][:simulation_steps]

    return centerline['innerBC'][:,:N], centerline['outerBC'][:,:N], centerline['c_lst'][:,:N]


class RaceTrack(object):
    def __init__(self):
        self.inner, self.outer, self.center = load_reference_track('c')
        self.N = self.center.shape[1]

        self.data = earcut.flatten([self.outer.T.tolist(), self.inner.T.tolist()])  
        self.triangles = earcut.earcut(self.data['vertices'], self.data['holes'], self.data['dimensions'])

        deviation = earcut.deviation(self.data['vertices'], self.data['holes'], self.data['dimensions'], self.triangles)
        assert(abs(deviation) < 1e-8)

        self.vertices_data = (gl.GLfloat * len(self.data['vertices']))(*self.data['vertices'])
        self.triangles_data = (gl.GLint * len(self.triangles))(*self.triangles)
        return

    def draw(self, magnify):
        """ Draws the race track

        Parameters
        ----------
        magnify : float
            the manification of the visualizer
        """
        gl.glColor3f(0.25,0.25,0.25)

        self.create_vertex_array_object(self.vertices_data, self.triangles_data)

        gl.glBindVertexArray(self.vertexArrayObject)
        gl.glDrawElements(gl.GL_TRIANGLES, len(self.triangles), gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        gl.glColor3f(0.5,0.5,0.5)
        gl.glLineWidth(0.3 / magnify)
        gl.glBegin(gl.GL_LINE_LOOP)
        for point in self.inner.T:
            gl.glVertex2f(*point)
        gl.glEnd()

        gl.glBegin(gl.GL_LINE_LOOP)
        for point in self.outer.T:
            gl.glVertex2f(*point)
        gl.glEnd()

        gl.glColor3f(1,1,1)
        gl.glLineWidth(0.005 / magnify)
        gl.glBegin(gl.GL_LINE_LOOP)
        for point in self.center.T:
            gl.glVertex2f(*point)
        gl.glEnd()
        
        gl.glColor3f(1,1,1)

    def create_vertex_array_object(self, vertex_array, index_array):
        self.vertexArrayObject = gl.GLuint(0)
        gl.glGenVertexArrays(1, self.vertexArrayObject)
        gl.glBindVertexArray(self.vertexArrayObject)

        ibo = gl.GLuint(0)
        gl.glGenBuffers(1, ibo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ibo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, 4*len(index_array), index_array, gl.GL_STATIC_DRAW)

        vbo = gl.GLuint(0)
        gl.glGenBuffers(1, vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 4*len(vertex_array), vertex_array, gl.GL_STATIC_DRAW)


        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 0, None)
        gl.glEnableVertexAttribArray(0)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

    def tracking_reward(self, C_ref, reference_x, reference_y, additional_reward=None):
        """ Returns a stage reward function for following the reference

        consists of control feature, velocity feature and optionally a feature for driving at the center of the road, i.e.
            cost = feature.control() - C_ref * (x - x_ref)**2 - C_ref * (y - y_ref)**2

        Parameters
        ----------
        C_ref : float
            parameter value for following the reference
        """
        params = cs.SX.sym('theta_human', 1, 1)
        reward = feature.control() + C_ref * feature.reference(reference_x, reference_y)
        reward_parametrized = feature.control() + params[0] * feature.reference(reference_x, reference_y)
            
        if additional_reward is not None:
            reward += additional_reward
            reward_parametrized += additional_reward
        return reward, reward_parametrized, params

    def get_reference(self, N, k):
        k_lap = k % self.N
        if k_lap + N >= self.N:
            return np.hstack([self.center[:,k_lap:], self.center[:,:(k_lap + N)%self.N]]).flatten().tolist()
        return self.center[:,k_lap:k_lap+N].flatten().tolist()
