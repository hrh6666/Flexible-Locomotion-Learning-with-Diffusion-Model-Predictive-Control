import numpy as np
from numpy.random import choice
from scipy import interpolate
from legged_gym.utils import trimesh

from isaacgym import terrain_utils, gymapi

import matplotlib.pyplot as plt

class TerrainStumbleMixAll:
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
        self.max_zScale=cfg.max_zScale
        self.min_zScale=cfg.min_zScale
        self.max_num_pit=cfg.max_num_pit
        self.min_num_pit=cfg.min_num_pit
        self.max_num_pole=cfg.max_num_pole
        self.min_num_pole=cfg.min_num_pole
        self.max_num_bar=cfg.max_num_bar
        self.min_num_bar=cfg.min_num_bar
        self.max_num_baffle = cfg.max_num_baffle
        self.min_num_baffle = cfg.min_num_baffle
        self.pit_length=cfg.pit_length
        self.pit_width=cfg.pit_width
        self.platform_size=cfg.platform_size
        self.border_width=cfg.border_width
        self.env_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3))
        self.heightsamples = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        print("Terrain heightsamples shape: ", self.heightsamples.shape)
        self.heightsamples_perlin= self.generate_fractal_noise_2d(self.xSize, self.ySize, self.tot_rows, self.tot_cols, **cfg.TerrainPerlin_kwargs)
        self.heightsamples_perlin = (self.heightsamples_perlin * (1 / cfg.vertical_scale)).astype(np.int16)
        
        self.max_square_pit_width = cfg.max_square_pit_width
        self.min_square_pit_width = cfg.min_square_pit_width
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
            noise += amplitude * TerrainStumbleMixAll.generate_perlin_noise_2d((xSamples, ySamples), (xScale, yScale)) * zScale
            amplitude *= fractalGain
            xScale, yScale = int(fractalLacunarity * xScale), int(fractalLacunarity * yScale)

        return noise
    
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                # if j<self.cfg.num_cols//2:
                if True:
                    terrain = terrain_utils.SubTerrain("terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels,
                                vertical_scale=self.vertical_scale,
                                horizontal_scale=self.horizontal_scale)

                    # bar阶段
                    if j < self.cfg.col_bar:
                        zScale = (self.min_zScale + j/(self.cfg.col_bar-1)*(self.max_zScale - self.min_zScale))
                        num_pit = 0
                        num_pole = 0
                        length = self.pit_length
                        width = self.min_square_pit_width + i/(self.cfg.num_rows-1)*(self.max_square_pit_width - self.min_square_pit_width)
                        platform_size = np.random.uniform(0.5*self.platform_size, 1.5*self.platform_size)
                        border_width = self.border_width
                        terrain = self.make_terrain_stumble(num_pit, num_pole, length, width, platform_size, border_width, zScale)

                    # baffle阶段（新增）
                    elif j >= self.cfg.col_bar and j < self.cfg.col_bar + self.cfg.col_baffle:
                        zScale = (self.min_zScale + (j - self.cfg.col_bar)/(self.cfg.num_rows-1)*(self.max_zScale - self.min_zScale))
                        num_pit = 0
                        num_pole = 0 
                        length = self.pit_length
                        width = self.min_square_pit_width + i/(self.cfg.num_rows-1)*(self.max_square_pit_width - self.min_square_pit_width)
                        platform_size = np.random.uniform(0.5*self.platform_size, 1.5*self.platform_size)
                        border_width = self.border_width
                        terrain = self.make_terrain_stumble(num_pit, num_pole, length, width, platform_size, border_width, zScale)

                    # pit阶段
                    elif j >= self.cfg.col_bar + self.cfg.col_baffle and j < self.cfg.col_bar + self.cfg.col_baffle + self.cfg.col_pit:
                        zScale = (self.min_zScale + (j - (self.cfg.col_bar + self.cfg.col_baffle))/(self.cfg.num_rows-1)*(self.max_zScale - self.min_zScale))
                        num_pit = 4
                        num_pole = 0
                        length = self.pit_length
                        width = self.min_square_pit_width + i/(self.cfg.num_rows-1)*(self.max_square_pit_width - self.min_square_pit_width)
                        platform_size = np.random.uniform(0.5*self.platform_size, 1.5*self.platform_size)
                        border_width = self.border_width
                        terrain = self.make_terrain_stumble(num_pit, num_pole, length, width, platform_size, border_width, zScale)

                    # stair阶段
                    elif j >= self.cfg.col_bar + self.cfg.col_baffle + self.cfg.col_pit and j < self.cfg.col_bar + self.cfg.col_baffle + self.cfg.col_pit + self.cfg.col_stair:
                        zScale = (self.min_zScale + (j - (self.cfg.col_bar + self.cfg.col_baffle + self.cfg.col_pit)) / (self.cfg.col_stair-1) * (self.max_zScale - self.min_zScale))
                        step_width = 0.30
                        step_height = -(self.cfg.min_step_height + i/(self.cfg.num_rows-1)*(self.cfg.max_step_height - self.cfg.min_step_height))
                        platform_size = np.random.uniform(0.8*self.platform_size, 1.2*self.platform_size)
                        border_width = self.border_width
                        terrain = self.make_terrain_stair(step_width, step_height, platform_size, border_width, zScale=0.05)

                    # slope阶段
                    elif j >= self.cfg.col_bar + self.cfg.col_baffle + self.cfg.col_pit + self.cfg.col_stair and j < self.cfg.col_bar + self.cfg.col_baffle + self.cfg.col_pit + self.cfg.col_stair + self.cfg.col_slope:
                        slope_degree = (self.cfg.min_height_degree + i/(self.cfg.num_rows-1)*(self.cfg.max_height_degree - self.cfg.min_height_degree))
                        platform_size = np.random.uniform(0.8*self.platform_size, 1.2*self.platform_size)
                        zScale = 0.05
                        border_width = self.border_width
                        terrain = self.make_terrain_slope(slope_degree, platform_size, border_width, zScale=zScale)

                    # pole阶段
                    elif j >= self.cfg.col_bar + self.cfg.col_baffle + self.cfg.col_pit + self.cfg.col_stair + self.cfg.col_slope and j < self.cfg.col_bar + self.cfg.col_baffle + self.cfg.col_pit + self.cfg.col_stair + self.cfg.col_slope + self.cfg.col_pole:
                        zScale = (self.min_zScale + (j - (self.cfg.col_bar + self.cfg.col_baffle + self.cfg.col_pit + self.cfg.col_stair + self.cfg.col_slope)) / (self.cfg.col_pole-1)*(self.max_zScale - self.min_zScale))
                        num_pit = 0
                        num_pole = round(self.min_num_pole + i/(self.cfg.num_rows-1)*(self.max_num_pole - self.min_num_pole))
                        length = self.pit_length
                        width = self.min_square_pit_width + i/(self.cfg.num_rows-1)*(self.max_square_pit_width - self.min_square_pit_width)
                        platform_size = np.random.uniform(0.5*self.platform_size, 1.5*self.platform_size)
                        border_width = self.border_width
                        terrain = self.make_terrain_stumble(num_pit, num_pole, length, width, platform_size, border_width, zScale)

                    # 其他
                    else:
                        zScale = (self.min_zScale + (j - (self.cfg.col_bar + self.cfg.col_baffle + self.cfg.col_pit + self.cfg.col_stair + self.cfg.col_slope + self.cfg.col_pole)) / (self.cfg.num_cols-1)*(self.max_zScale - self.min_zScale))
                        num_pit = 0
                        num_pole = 0
                        length = self.pit_length
                        width = self.min_square_pit_width + i/(self.cfg.num_rows-1)*(self.max_square_pit_width - self.min_square_pit_width)
                        platform_size = np.random.uniform(0.5*self.platform_size, 1.5*self.platform_size)
                        border_width = self.border_width
                        terrain = self.make_terrain_stumble(num_pit, num_pole, length, width, platform_size, border_width, zScale)

                self.add_terrain_to_map(terrain, i, j)

    def make_terrain_stumble(self, num_pit, num_pole, length, width, platform_size, border_width, zScale=0.05):
        terrain = terrain_utils.SubTerrain("terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels,
                                vertical_scale=self.vertical_scale,
                                horizontal_scale=self.horizontal_scale)
        self.stumble_terrain_border(terrain, num_pit, num_pole, length, width, platform_size, border_width, zScale)
        return terrain
    
    def make_terrain_stair(self,step_width,step_height,platform_size,border_width,zScale=0.05):
        terrain = terrain_utils.SubTerrain("terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels,
                                vertical_scale=self.vertical_scale,
                                horizontal_scale=self.horizontal_scale)
        self.pyramid_stairs_terrain_border(terrain, step_width=step_width, step_height=step_height,
                                           platform_size=platform_size,border_width=border_width,zScale=zScale)
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

    def stumble_terrain_border(self, terrain, num_pit, num_pole, length, width, platform_size, border_width, zScale=0.05):
        (i, j) = terrain.height_field_raw.shape
        border_width_ = round(border_width/terrain.horizontal_scale)
        platform_size_ = round(platform_size/terrain.horizontal_scale)
        width_ = round(np.random.uniform(width*0.5, width*1.5) / terrain.horizontal_scale)
        length_ = round(np.random.uniform(length*0.5, length*1.5) / terrain.horizontal_scale)
        height_ = round(np.random.uniform(1.0, 2.0)/terrain.vertical_scale)
        width_save = width_
        length_save = length_
        #print("num_pit", num_pit)
        for k in range(num_pit):
            if k == 0:
                terrain.height_field_raw[i // 2 + length_save // 2  - width_ // 2 : i // 2 + length_save // 2 + (width_+1) // 2, j // 2 - length_ // 2 : j // 2 + length_ // 2] = -height_
            elif k == 1:
                terrain.height_field_raw[i // 2 - length_save // 2   : i // 2 + length_save // 2, j // 2 + length_ // 2 - width_ // 2 : j // 2 + length_ // 2 + (width_+1) // 2] = -height_
            elif k == 2:
                terrain.height_field_raw[i // 2 - length_save // 2  - width_ // 2 : i // 2 - length_save // 2 + (width_+1) // 2, j // 2 - length_ // 2 : j // 2 + length_ // 2] = -height_
            else:
                terrain.height_field_raw[i // 2 - length_save // 2   : i // 2 + length_save // 2, j // 2 - length_ // 2 - width_ // 2 : j // 2 - length_ // 2 + (width_+1) // 2] = -height_
            

        for _ in range(num_pole):
            width_ = round(np.random.uniform(0.05, 0.1)/terrain.horizontal_scale)
            height_ = round(np.random.uniform(0.5, 1.0)/terrain.vertical_scale)

            start_i = round(np.random.uniform(border_width_, i-border_width_))
            start_j = round(np.random.uniform(border_width_, j-border_width_))
            terrain.height_field_raw[start_i:start_i+width_, start_j:start_j+width_] = height_

        x1 = (terrain.width - platform_size_) // 2
        x2 = (terrain.width + platform_size_) // 2
        y1 = (terrain.length - platform_size_) // 2
        y2 = (terrain.length + platform_size_) // 2
        terrain.height_field_raw[x1:x2, y1:y2] = 0

        heightsamples=self.generate_fractal_noise_2d(self.env_length, self.env_width, self.length_per_env_pixels, self.width_per_env_pixels, frequency=10, zScale=zScale)
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
    
    def add_trimesh_to_sim(self, gym, sim, trimesh):
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = trimesh[0].shape[0]
        tm_params.nb_triangles = trimesh[1].shape[0]
        # tm_params.transform.p.x = trimesh_origin[0]
        # tm_params.transform.p.y = trimesh_origin[1]
        # tm_params.transform.p.z = trimesh_origin[2]
        tm_params.static_friction = self.cfg.static_friction
        tm_params.dynamic_friction = self.cfg.dynamic_friction
        tm_params.restitution = self.cfg.restitution
        gym.add_triangle_mesh(
            sim,
            trimesh[0].flatten(order= "C"),
            trimesh[1].flatten(order= "C"),
            tm_params,
        )

    def add_obstacle_to_sim(self, gym, sim, device="cpu"):
        start_of_bar = 0
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                if j>=start_of_bar and j < start_of_bar + self.cfg.col_bar:
                # if True:
                    if round(self.min_num_bar+i/(self.cfg.num_rows-1)*(self.max_num_bar-self.min_num_bar)) != 0:
                        num_bar = 4
                    else:
                        num_bar = 0
                    width = np.random.uniform(self.pit_width*0.01, self.pit_width*0.5)
                    length = np.random.uniform(self.pit_length*0.8, self.pit_length*1.8)
                    height_ = 0.+i/(self.cfg.num_rows-1)*(0.2-0.)

                    for k in range(num_bar):
                        width_=width
                        length_=length
                        width_save = width_
                        if k == 1 or k == 3:
                            length_, width_ = width_, length_

                        size=np.array([width_, length_, width_save], dtype= np.float32)
                        if k == 0:
                            center=np.array([length/2, 0, height_], dtype= np.float32)+self.env_origins[i, j]
                        elif k == 1:
                            center=np.array([0, length/2, height_], dtype= np.float32)+self.env_origins[i, j]
                        elif k == 2:
                            center=np.array([-length/2, 0, height_], dtype= np.float32)+self.env_origins[i, j]
                        else:
                            center=np.array([0, -length/2, height_], dtype= np.float32)+self.env_origins[i, j]
                        center=np.array(center, dtype=np.float32)
                        
                        obstacle_trimesh = trimesh.box_trimesh(size, center)
                        self.add_trimesh_to_sim(gym, sim, obstacle_trimesh)
        self.add_baffle_to_sim(gym=gym, sim=sim, device = device)

    def add_baffle_to_sim(self, gym, sim, device="cpu"):
        start_of_baffle = self.cfg.col_bar
        start_height = 0.32
        end_height = 0.19
        height_diff = start_height - end_height
        
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                if j>=start_of_baffle and j < start_of_baffle + self.cfg.col_baffle:
                    if round(self.min_num_baffle+i/(self.cfg.num_rows-1)*(self.max_num_baffle-self.min_num_baffle)) != 0:
                        num_baffle = 4
                    else:
                        num_baffle = 0
                    width = 0.3
                    length = np.random.uniform(self.pit_length*0.8, self.pit_length*1.8)
                    height_ = start_height - (i / (self.cfg.num_rows - 1)) * height_diff + width / 2

                    for k in range(num_baffle):
                        width_=0.1
                        length_=length
                        width_save = width
                        if k == 1 or k == 3:
                            length_, width_ = width_, length_

                        size=np.array([width_, length_, width_save], dtype= np.float32)
                        if k == 0:
                            center=np.array([length/2, 0, height_], dtype= np.float32)+self.env_origins[i, j]
                        elif k == 1:
                            center=np.array([0, length/2, height_], dtype= np.float32)+self.env_origins[i, j]
                        elif k == 2:
                            center=np.array([-length/2, 0, height_], dtype= np.float32)+self.env_origins[i, j]
                        else:
                            center=np.array([0, -length/2, height_], dtype= np.float32)+self.env_origins[i, j]
                        center=np.array(center, dtype=np.float32)
                        
                        obstacle_trimesh = trimesh.box_trimesh(size, center)
                        self.add_trimesh_to_sim(gym, sim, obstacle_trimesh)
    

    def add_terrain_to_sim(self, gym, sim, device= "cpu"):
        """ deploy the terrain mesh to the simulator
        """
        self._create_trimesh(gym, sim, device)
        self.add_obstacle_to_sim(gym, sim, device)
