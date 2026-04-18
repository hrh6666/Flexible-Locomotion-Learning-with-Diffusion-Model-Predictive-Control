import importlib

from legged_gym.utils.terrain.multitask_terrain import TerrainMultitask
from legged_gym.utils.terrain.stumble_mixall import TerrainStumbleMixAll

terrain_registry = dict(
    Terrain= "legged_gym.utils.terrain.terrain:Terrain",
    BarrierTrack= "legged_gym.utils.terrain.barrier_track:BarrierTrack",
    TerrainPerlin= "legged_gym.utils.terrain.perlin:TerrainPerlin",
    TerrainMix= "legged_gym.utils.terrain.mix:TerrainMix",
    TerrainMixOld= "legged_gym.utils.terrain.mixold:TerrainMixOld",
    TerrainStair= "legged_gym.utils.terrain.stair:TerrainStair",
    TerrainStairUp= "legged_gym.utils.terrain.stairup:TerrainStairUp",
    TerrainStairDown= "legged_gym.utils.terrain.stairdown:TerrainStairDown",
    TerrainSlope= "legged_gym.utils.terrain.slope:TerrainSlope",
    TerrainSlopeUp= "legged_gym.utils.terrain.slopeup:TerrainSlopeUp",
    TerrainSlopeDown= "legged_gym.utils.terrain.slopedown:TerrainSlopeDown",
    TerrainWave= "legged_gym.utils.terrain.wave:TerrainWave",
    TerrainObstacle= "legged_gym.utils.terrain.obstacle:TerrainObstacle",
    TerrainPlane= "legged_gym.utils.terrain.plane:TerrainPlane",
    TerrainGap= "legged_gym.utils.terrain.gap:TerrainGap",
    TerrainBridge= "legged_gym.utils.terrain.bridge:TerrainBridge",
    TerrainStumble= "legged_gym.utils.terrain.stumble:TerrainStumble",
    TerrainStumbleMix= "legged_gym.utils.terrain.stumble_mix:TerrainStumbleMix",
    TerrainStumbleOld= "legged_gym.utils.terrain.stumble_old:TerrainStumbleOld",
    TerrainStumbleSquare= "legged_gym.utils.terrain.stumble_square:TerrainStumbleSquare",
    TerrainStumbleBarTrack = "legged_gym.utils.terrain.stumble_bar_track:TerrainStumbleBarTrack",
    TerrainStumbleTrackTest = "legged_gym.utils.terrain.stumble_track_test:TerrainStumbleTrackTest",
    TerrainStumbleMixAll = "legged_gym.utils.terrain.stumble_mixall:TerrainStumbleMixAll",
    TerrainMultitask = "legged_gym.utils.terrain.multitask_terrain:TerrainMultitask",
    TerrainMultitaskTrack = "legged_gym.utils.terrain.multitask_track:TerrainMultitaskTrack",
)

def get_terrain_cls(terrain_cls):
    entry_point = terrain_registry[terrain_cls]
    module, class_name = entry_point.rsplit(":", 1)
    module = importlib.import_module(module)
    return getattr(module, class_name)
