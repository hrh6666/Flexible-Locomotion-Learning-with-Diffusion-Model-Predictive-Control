import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils, gymapi

import matplotlib.pyplot as plt

class TerrainWave:
    def __init__(self, cfg, num_envs):
        self.cfg = cfg
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.xSize = cfg.terrain_length * cfg.num_rows # int(cfg.horizontal_scale * cfg.tot_cols)
        self.ySize = cfg.terrain_width * cfg.num_cols # int(cfg.horizontal_scale * cfg.tot_rows)
        self.vertical_scale = cfg.vertical_scale
        self.horizontal_scale = cfg.horizontal_scale
        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)
        self.tot_cols = int(self.ySize / cfg.horizontal_scale)
        self.tot_rows = int(self.xSize / cfg.horizontal_scale)
        self.max_step_height = cfg.max_step_height
        self.min_step_height = cfg.min_step_height
        self.max_step_width = cfg.max_step_width
        self.min_step_width = cfg.min_step_width
        self.max_platform_size = cfg.max_platform_size
        self.min_platform_size = cfg.min_platform_size
        self.max_height_degree = cfg.max_height_degree
        self.min_height_degree = cfg.min_height_degree
        self.min_wave_amplitude=cfg.min_wave_amplitude
        self.max_wave_amplitude=cfg.max_wave_amplitude
        self.min_obstacle_height=cfg.min_obstacle_height
        self.max_obstacle_height=cfg.max_obstacle_height
        self.max_zScale=cfg.max_zScale
        self.min_zScale=cfg.min_zScale
        self.border_width = cfg.border_width
        self.env_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3))
        self.heightsamples = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        print("Terrain heightsamples shape: ", self.heightsamples.shape)
        self.heightsamples_perlin= self.generate_fractal_noise_2d(self.xSize, self.ySize, self.tot_rows, self.tot_cols, **cfg.TerrainPerlin_kwargs)
        self.heightsamples_perlin = (self.heightsamples_perlin * (1 / cfg.vertical_scale)).astype(np.int16)
        # self.heightsamples_perlin[:,:self.heightsamples_perlin.shape[1]//4]=0
        # self.heightsamples_perlin[:,self.heightsamples_perlin.shape[1]//2:3*self.heightsamples_perlin.shape[1]//4]=0
        self.curiculum()

        # self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.heightsamples+self.heightsamples_perlin,
        self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.heightsamples,
                                                                                        cfg.horizontal_scale,
                                                                                        cfg.vertical_scale,
                                                                                        cfg.slope_treshold)
    
    @staticmethod
    def generate_perlin_noise_2d(shape, res):
        def f(t):
            return 6*t**5 - 15*t**4 + 10*t**3

        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
        # Gradients
        angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
        g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
        # Ramps
        n00 = np.sum(grid * g00, 2)
        n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
        n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
        # Interpolation
        t = f(grid)
        n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
        n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
        return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1) * 0.5 + 0.5
    
    @staticmethod
    def generate_fractal_noise_2d(xSize=20, ySize=20, xSamples=1600, ySamples=1600, \
        frequency=10, fractalOctaves=2, fractalLacunarity = 2.0, fractalGain=0.25, zScale = 0.23):
        xScale = int(frequency * xSize)
        yScale = int(frequency * ySize)
        amplitude = 1
        shape = (xSamples, ySamples)
        noise = np.zeros(shape)
        for _ in range(fractalOctaves):
            noise += amplitude * TerrainWave.generate_perlin_noise_2d((xSamples, ySamples), (xScale, yScale)) * zScale
            amplitude *= fractalGain
            xScale, yScale = int(fractalLacunarity * xScale), int(fractalLacunarity * yScale)

        return noise
    
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                # if j<self.cfg.num_cols//2:
                # if True:
                if (j%2)==0:
                    terrain = terrain_utils.SubTerrain("terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels,
                                vertical_scale=self.vertical_scale,
                                horizontal_scale=self.horizontal_scale)
                    wave_amplitude=(self.min_wave_amplitude+i/(self.cfg.num_rows-1)*(self.max_wave_amplitude-self.min_wave_amplitude))
                    terrain=terrain_utils.wave_terrain(terrain, num_waves=3, amplitude = wave_amplitude)
                    heightsamples=self.generate_fractal_noise_2d(self.env_length, self.env_width, self.length_per_env_pixels, self.width_per_env_pixels, frequency=5, zScale=0.05)
                    heightsamples = (heightsamples * (1 / self.cfg.vertical_scale)).astype(np.int16)
                    terrain.height_field_raw+=heightsamples
                elif (j%2)==1:
                    terrain = terrain_utils.SubTerrain("terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels,
                                vertical_scale=self.vertical_scale,
                                horizontal_scale=self.horizontal_scale)
                    wave_amplitude=(self.min_wave_amplitude+i/(self.cfg.num_rows-1)*(self.max_wave_amplitude-self.min_wave_amplitude))
                    terrain=terrain_utils.wave_terrain(terrain, num_waves=4, amplitude = wave_amplitude)
                    heightsamples=self.generate_fractal_noise_2d(self.env_length, self.env_width, self.length_per_env_pixels, self.width_per_env_pixels, frequency=5, zScale=0.05)
                    heightsamples = (heightsamples * (1 / self.cfg.vertical_scale)).astype(np.int16)
                    terrain.height_field_raw+=heightsamples
                self.add_terrain_to_map(terrain, i, j) 
                
    # def sloped_terrain(self, terrain, slope=1, border=1.):
    #     """
    #     Generate a sloped terrain

    #     Parameters:
    #         terrain (SubTerrain): the terrain
    #         slope (int): positive or negative slope
    #     Returns:
    #         terrain (SubTerrain): update terrain
    #     """

    #     height_pix=int(slope / terrain.vertical_scale)
    #     border_width = int(border/terrain.horizon_scale)
        
    #     for x in range(self.width_per_env_pixels):
    #         if x<border_width:
    #             terrain.height_field_raw[x, :] = 0
    #         elif x>=border_width and x<(self.width_per_env_pixels-border_width)/2:
    #             terrain.height_field_raw[x, :] = 
    #     return terrain

    
    def make_terrain_stair(self,step_width,step_height,platform_size,border_width,zScale=0.05):
        terrain = terrain_utils.SubTerrain("terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels,
                                vertical_scale=self.vertical_scale,
                                horizontal_scale=self.horizontal_scale)
        self.pyramid_stairs_terrain_border(terrain, step_width=step_width, step_height=step_height,
                                           platform_size=platform_size,border_width=border_width,zScale=zScale)
        return terrain
    
    def make_terrain_perlin(self,zScale):
        heightsamples = np.zeros(( self.length_per_env_pixels, self.width_per_env_pixels), dtype=np.int16)
        terrain = terrain_utils.SubTerrain("terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels,
                                vertical_scale=self.vertical_scale,
                                horizontal_scale=self.horizontal_scale)
        heightsamples=self.generate_fractal_noise_2d(self.env_length, self.env_width, self.length_per_env_pixels, self.width_per_env_pixels, frequency=5, zScale=zScale)
        heightsamples = (heightsamples * (1 / self.cfg.vertical_scale)).astype(np.int16)
        terrain.height_field_raw=heightsamples
        return terrain
    
    def make_terrain_slope(self,slope_degree,platform_size,border_width,zScale=0.05):
        terrain = terrain_utils.SubTerrain("terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels,
                                vertical_scale=self.vertical_scale,
                                horizontal_scale=self.horizontal_scale)
        self.pyramid_slope_terrain_border(terrain, slope_degree=slope_degree, 
                                           platform_size=platform_size,border_width=border_width,zScale=zScale)
        return terrain
    
    def pyramid_stairs_terrain_border(self, terrain, step_width, step_height, platform_size, border_width,zScale=0.05):
        # switch parameters to discrete units
        step_width = round(step_width / terrain.horizontal_scale)
        step_height = round(step_height / terrain.vertical_scale)
        platform_size = round(platform_size / terrain.horizontal_scale)
        border_size = round(border_width / terrain.horizontal_scale)

        height = 0
        start_x = border_size
        stop_x = terrain.width-border_size
        start_y = border_size
        stop_y = terrain.length-border_size
        while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
            terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height
            start_x += step_width
            stop_x -= step_width
            start_y += step_width
            stop_y -= step_width
            height += step_height
        
        # terrain.height_field_raw[:,:]=2/terrain.vertical_scale
        # terrain.height_field_raw[:start_x+step_width,:]=0
        # terrain.height_field_raw[stop_x-step_width:,:]=0
        # while (stop_x - start_x) > platform_size:
        #     start_x += step_width
        #     stop_x -= step_width
        #     start_y += step_width
        #     stop_y -= step_width
        #     height += step_height
        #     terrain.height_field_raw[start_x: start_x+step_width, start_y: stop_y] = height
        #     terrain.height_field_raw[stop_x-step_width: stop_x, start_y: stop_y] = height
        # terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height
        heightsamples=self.generate_fractal_noise_2d(self.env_length, self.env_width, self.length_per_env_pixels, self.width_per_env_pixels, frequency=5, zScale=zScale)
        heightsamples = (heightsamples * (1 / self.cfg.vertical_scale)).astype(np.int16)
        terrain.height_field_raw+=heightsamples
        
            
        return terrain
    
    def pyramid_slope_terrain_border(self, terrain, slope_degree, platform_size, border_width,zScale=0.05):
        # switch parameters to discrete units
        step_height = round(terrain.horizontal_scale*np.tan(slope_degree) / terrain.vertical_scale)
        platform_size = round(platform_size / terrain.horizontal_scale)
        border_size = round(border_width / terrain.horizontal_scale)

        height = 0
        start_x = border_size
        stop_x = terrain.width-border_size
        start_y = border_size
        stop_y = terrain.length-border_size
        while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
            terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height
            start_x += 1
            stop_x -= 1
            start_y += 1
            stop_y -= 1
            height += step_height
        
        # terrain.height_field_raw[:,:]=2/terrain.vertical_scale
        # terrain.height_field_raw[:start_x+step_width,:]=0
        # terrain.height_field_raw[stop_x-step_width:,:]=0
        # while (stop_x - start_x) > platform_size:
        #     start_x += step_width
        #     stop_x -= step_width
        #     start_y += step_width
        #     stop_y -= step_width
        #     height += step_height
        #     terrain.height_field_raw[start_x: start_x+step_width, start_y: stop_y] = height
        #     terrain.height_field_raw[stop_x-step_width: stop_x, start_y: stop_y] = height
        # terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height
        heightsamples=self.generate_fractal_noise_2d(self.env_length, self.env_width, self.length_per_env_pixels, self.width_per_env_pixels, frequency=5, zScale=zScale)
        heightsamples = (heightsamples * (1 / self.cfg.vertical_scale)).astype(np.int16)
        terrain.height_field_raw+=heightsamples
        
            
        return terrain    

        
    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = i * self.length_per_env_pixels
        end_x = (i + 1) * self.length_per_env_pixels
        start_y = j * self.width_per_env_pixels
        end_y = (j + 1) * self.width_per_env_pixels
        self.heightsamples[start_x: end_x, start_y: end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x0 = int(self.env_length/2 / terrain.horizontal_scale)
        y0 = int(self.env_width/2 / terrain.horizontal_scale)
        env_origin_z = terrain.height_field_raw[x0, y0]*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        
    def _create_trimesh(self, gym, sim, device= "cpu"):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        trimesh=(self.vertices, self.triangles)
        trimesh_origin=np.zeros(3,)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = trimesh[0].shape[0]
        tm_params.nb_triangles = trimesh[1].shape[0]
        tm_params.transform.p.x = trimesh_origin[0]
        tm_params.transform.p.y = trimesh_origin[1]
        tm_params.transform.p.z = 0.
        tm_params.static_friction = self.cfg.static_friction
        tm_params.dynamic_friction = self.cfg.dynamic_friction
        tm_params.restitution = self.cfg.restitution
        gym.add_triangle_mesh(
            sim,
            trimesh[0].flatten(order= "C"),
            trimesh[1].flatten(order= "C"),
            tm_params,
        )

    def add_terrain_to_sim(self, gym, sim, device= "cpu"):
        """ deploy the terrain mesh to the simulator
        """
        self._create_trimesh(gym, sim, device)