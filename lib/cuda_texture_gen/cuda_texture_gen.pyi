import collections.abc
import enum
import nanobind
from _typeshed import Incomplete
from typing import Callable, ClassVar

blur: nanobind.nb_func
cuda_hello: nanobind.nb_func
generate_ao_map: nanobind.nb_func
generate_normal_map: nanobind.nb_func
print_debug_info: nanobind.nb_func

class DeviceArrayFloat1D:
    array: Incomplete
    def __init__(self) -> None: ...
    def dev_ptr(self) -> int: ...
    def resize(self, arg: collections.abc.Sequence[int]) -> None: ...
    def size(self) -> int: ...
    def __copy__(self) -> DeviceArrayFloat1D: ...
    def __deepcopy__(self, arg: dict) -> DeviceArrayFloat1D: ...
    @property
    def shape(self) -> list[int]: ...

class DeviceArrayFloat2D:
    array: Incomplete
    def __init__(self) -> None: ...
    def dev_ptr(self) -> int: ...
    def resize(self, arg: collections.abc.Sequence[int]) -> None: ...
    def size(self) -> int: ...
    def __copy__(self) -> DeviceArrayFloat2D: ...
    def __deepcopy__(self, arg: dict) -> DeviceArrayFloat2D: ...
    @property
    def shape(self) -> list[int]: ...

class DeviceArrayFloat3D:
    array: Incomplete
    def __init__(self) -> None: ...
    def dev_ptr(self) -> int: ...
    def resize(self, arg: collections.abc.Sequence[int]) -> None: ...
    def size(self) -> int: ...
    def __copy__(self) -> DeviceArrayFloat3D: ...
    def __deepcopy__(self, arg: dict) -> DeviceArrayFloat3D: ...
    @property
    def shape(self) -> list[int]: ...

class DeviceArrayFloat4D:
    array: Incomplete
    def __init__(self) -> None: ...
    def dev_ptr(self) -> int: ...
    def resize(self, arg: collections.abc.Sequence[int]) -> None: ...
    def size(self) -> int: ...
    def __copy__(self) -> DeviceArrayFloat4D: ...
    def __deepcopy__(self, arg: dict) -> DeviceArrayFloat4D: ...
    @property
    def shape(self) -> list[int]: ...

class DeviceArrayInt1D:
    array: Incomplete
    def __init__(self) -> None: ...
    def dev_ptr(self) -> int: ...
    def resize(self, arg: collections.abc.Sequence[int]) -> None: ...
    def size(self) -> int: ...
    def __copy__(self) -> DeviceArrayInt1D: ...
    def __deepcopy__(self, arg: dict) -> DeviceArrayInt1D: ...
    @property
    def shape(self) -> list[int]: ...

class DeviceArrayInt2D:
    array: Incomplete
    def __init__(self) -> None: ...
    def dev_ptr(self) -> int: ...
    def resize(self, arg: collections.abc.Sequence[int]) -> None: ...
    def size(self) -> int: ...
    def __copy__(self) -> DeviceArrayInt2D: ...
    def __deepcopy__(self, arg: dict) -> DeviceArrayInt2D: ...
    @property
    def shape(self) -> list[int]: ...

class DeviceArrayInt3D:
    array: Incomplete
    def __init__(self) -> None: ...
    def dev_ptr(self) -> int: ...
    def resize(self, arg: collections.abc.Sequence[int]) -> None: ...
    def size(self) -> int: ...
    def __copy__(self) -> DeviceArrayInt3D: ...
    def __deepcopy__(self, arg: dict) -> DeviceArrayInt3D: ...
    @property
    def shape(self) -> list[int]: ...

class DeviceArrayInt4D:
    array: Incomplete
    def __init__(self) -> None: ...
    def dev_ptr(self) -> int: ...
    def resize(self, arg: collections.abc.Sequence[int]) -> None: ...
    def size(self) -> int: ...
    def __copy__(self) -> DeviceArrayInt4D: ...
    def __deepcopy__(self, arg: dict) -> DeviceArrayInt4D: ...
    @property
    def shape(self) -> list[int]: ...

class Erosion10:
    _block: int
    _debug: bool
    _debug_drain_total: float
    _debug_erosion_total: float
    _debug_evaporation_total: float
    _debug_rain_total: float
    _exposed_layer_map: Incomplete
    _flux8: Incomplete
    _height: int
    _layers: int
    _main_loop: bool
    _sea_map: Incomplete
    _sediment_flux8: Incomplete
    _sediment_out: Incomplete
    _slope_magnitude: Incomplete
    _slope_vector2: Incomplete
    _step: int
    _water_out: Incomplete
    _water_velocity: Incomplete
    _width: int
    _wind_vector2: Incomplete
    debug_mod: int
    debug_print: bool
    deposition_mode: int
    deposition_rate: float
    deposition_threshold: float
    drain_rate: float
    erosion_mode: int
    erosion_rate: float
    evaporation_mode: int
    evaporation_rate: float
    flow_rate: float
    hardness_map: Incomplete
    height_map: Incomplete
    layer_erosion_threshold: Incomplete
    layer_erosiveness: Incomplete
    layer_map: Incomplete
    layer_permeability: Incomplete
    layer_solubility: Incomplete
    layer_yield: Incomplete
    max_height: float
    max_water_outflow: float
    min_height: float
    mode: int
    rain_map: Incomplete
    rain_rate: float
    scale: float
    sea_level: float
    sea_pass: bool
    sea_tidal_range: float
    sediment_capacity: float
    sediment_layer_map: Incomplete
    sediment_layer_mode: bool
    sediment_map: Incomplete
    sediment_yield: float
    simple_collapse: bool
    simple_collapse_amount: float
    simple_collapse_jitter: float
    simple_collapse_threshold: float
    simple_collapse_yield: float
    slope_jitter: float
    slope_jitter_mode: int
    steps: int
    water_map: Incomplete
    wind_strength: float
    wrap: bool
    def __init__(self) -> None: ...
    def allocate_device(self) -> None: ...
    def debug_update(self) -> None: ...
    def process(self) -> None: ...

class Erosion9:
    _block: int
    _calculation_time: float
    _debug: bool
    _debug_drain_total: float
    _debug_erosion_total: float
    _debug_evaporation_total: float
    _debug_rain_total: float
    _flux8: Incomplete
    _height: int
    _height_map_out: Incomplete
    _layer_map_out: Incomplete
    _layers: int
    _sediment_flux8: Incomplete
    _sediment_map_out: Incomplete
    _slope_map: Incomplete
    _surface_map: Incomplete
    _water_map_out: Incomplete
    _width: int
    correct_diagonal_distance: bool
    debug_mod: int
    debug_print: bool
    deposition_mode: int
    deposition_rate: float
    deposition_threshold: float
    diffusion_rate: float
    drain_rate: float
    erosion_mode: int
    erosion_rate: float
    evaporation_rate: float
    flow_rate: float
    gravity: float
    hardness_map: Incomplete
    height_map: Incomplete
    layer_map: Incomplete
    layers_erosiveness: list[float]
    layers_permeability: list[float]
    layers_threshold: list[float]
    layers_yield: list[float]
    manning_mode: int
    max_height: float
    max_water_outflow: float
    min_height: float
    mode: int
    outflow_carve: float
    positive_slope_gradient_cap: float
    rain_map: Incomplete
    rain_random: bool
    rain_rate: Incomplete
    scale: float
    sediment_capacity: float
    sediment_drain_rate: float
    sediment_map: Incomplete
    sediment_yield: float
    simple_erosion_rate: float
    slope_exponent: float
    slope_jitter: float
    slope_jitter_mode: int
    slope_threshold: float
    steps: int
    water_map: Incomplete
    wrap: bool
    def __init__(self) -> None: ...
    def allocate_device(self) -> None: ...
    def debug_update(self) -> None: ...
    def process(self) -> None: ...

class GNC_Erosion:
    _debug: Incomplete
    _exposed_layer_map: Incomplete
    _flux8_map: Incomplete
    _layer_count: Incomplete
    _layer_mode: Incomplete
    _sea_map: Incomplete
    _sediment_flux8_map: Incomplete
    _sediment_out_map: Incomplete
    _size: Incomplete
    _slope_vector2_map: Incomplete
    _step: Incomplete
    _water_out_map: Incomplete
    _water_velocity_map: Incomplete
    _wind_vector2_map: Incomplete
    debug_mod: Incomplete
    debug_print: Incomplete
    deposition_mode: Incomplete
    deposition_rate: Incomplete
    deposition_threshold: Incomplete
    drain_rate: Incomplete
    erosion_mode: Incomplete
    erosion_rate: Incomplete
    evaporation_mode: Incomplete
    evaporation_rate: Incomplete
    flow_rate: Incomplete
    height_map: Incomplete
    layer_erosion_threshold_array: Incomplete
    layer_erosiveness_array: Incomplete
    layer_map: Incomplete
    layer_permeability_array: Incomplete
    layer_solubility_array: Incomplete
    layer_yield_array: Incomplete
    max_height: Incomplete
    max_water_outflow: Incomplete
    min_height: Incomplete
    rain_map: Incomplete
    rain_rate: Incomplete
    scale: Incomplete
    sea_level: Incomplete
    sea_pass: Incomplete
    sea_tidal_range: Incomplete
    sediment_capacity: Incomplete
    sediment_layer_map: Incomplete
    sediment_layer_mode: Incomplete
    sediment_map: Incomplete
    sediment_yield: Incomplete
    simple_collapse: Incomplete
    simple_collapse_amount: Incomplete
    simple_collapse_jitter: Incomplete
    simple_collapse_threshold: Incomplete
    simple_collapse_yield: Incomplete
    slope_jitter: Incomplete
    slope_jitter_mode: Incomplete
    steps: Incomplete
    stream: Incomplete
    water_map: Incomplete
    wind_strength: Incomplete
    wrap: Incomplete
    def __init__(self) -> None: ...
    def compute(self) -> None: ...
    def process(self) -> None: ...
    def setup(self) -> None: ...

class GNC_Erosion2:
    _exposed_layer_map: Incomplete
    _layer_count: Incomplete
    _layer_mode: Incomplete
    _size: Incomplete
    _slope_vector2_map: Incomplete
    _step: Incomplete
    height_map: Incomplete
    layer_map: Incomplete
    rain_map: Incomplete
    sediment_map: Incomplete
    slope_jitter: Incomplete
    steps: Incomplete
    stream: Incomplete
    water_map: Incomplete
    wrap: Incomplete
    def __init__(self) -> None: ...
    def compute(self) -> None: ...
    def process(self) -> None: ...
    def setup(self) -> None: ...

class GNC_Example:
    _debug: Incomplete
    _height: Incomplete
    _width: Incomplete
    extra_test: Incomplete
    input: Incomplete
    output: Incomplete
    stream: Incomplete
    tile_size: Incomplete
    def __init__(self) -> None: ...
    def compute(self) -> None: ...
    def process(self) -> None: ...

class GNC_Noise:
    _debug: Incomplete
    basis3: Incomplete
    offset: Incomplete
    output: Incomplete
    period: Incomplete
    rotation: Incomplete
    seed: Incomplete
    size: Incomplete
    smoothing_mode: Incomplete
    stream: Incomplete
    wrap: Incomplete
    def __init__(self) -> None: ...
    def compute(self) -> None: ...
    def process(self) -> None: ...

class GNC_Resample:
    _size: Incomplete
    input: Incomplete
    map_x: Incomplete
    map_y: Incomplete
    output: Incomplete
    relative_offset: Incomplete
    sample_mode: Incomplete
    scale_by_output_size: Incomplete
    stream: Incomplete
    def __init__(self) -> None: ...
    def compute(self) -> None: ...
    def process(self) -> None: ...

class GNC_SlopeErosion:
    _size: Incomplete
    _step: Incomplete
    deposition_rate: Incomplete
    erosion_rate: Incomplete
    height_map: Incomplete
    jitter: Incomplete
    sediment_map: Incomplete
    slope_threshold: Incomplete
    steps: Incomplete
    stream: Incomplete
    wrap: Incomplete
    def __init__(self) -> None: ...
    def compute(self) -> None: ...
    def process(self) -> None: ...

class GNC_Template:
    _debug: Incomplete
    _size: Incomplete
    float8: Incomplete
    input: Incomplete
    input2: Incomplete
    int8: Incomplete
    output: Incomplete
    output2: Incomplete
    stream: Incomplete
    tile_size: Incomplete
    def __init__(self) -> None: ...
    def compute(self) -> None: ...
    def process(self) -> None: ...
    def test(self) -> None: ...

class GNC_Wind:
    _map_size: Incomplete
    _step: Incomplete
    damp_wind: Incomplete
    dust_map: Incomplete
    height_map: Incomplete
    random_wind: Incomplete
    slope_influence: Incomplete
    slope_vec2_map: Incomplete
    stream: Incomplete
    test_float2: Incomplete
    wind_drag: Incomplete
    wind_influence: Incomplete
    wind_vec2_map: Incomplete
    wind_vec2_map_out: Incomplete
    wrap: Incomplete
    def __init__(self) -> None: ...
    def compute(self) -> None: ...
    def process(self) -> None: ...

class GraphNode:
    output: Incomplete
    def __init__(self) -> None: ...
    def connect_input(self, output_node: GraphNode, output_port: int, input_port: int) -> bool: ...
    def process(self) -> None: ...

class Noise3D:
    _block: int
    _scale: float
    _scale_x: float
    _scale_y: float
    _scale_z: float
    height: int
    period: float
    period_x: float
    period_y: float
    period_z: float
    rotate_x: float
    rotate_y: float
    rotate_z: float
    seed: Incomplete
    type: Incomplete
    width: int
    wrap_x: Incomplete
    wrap_y: Incomplete
    wrap_z: Incomplete
    x: float
    y: float
    z: float
    def __init__(self) -> None: ...
    @property
    def noise(self): ...

class NoiseGenerator:
    class Type(enum.Enum):
        __new__: ClassVar[Callable] = ...
        Gradient2D: ClassVar[NoiseGenerator.Type] = ...
        Gradient3D: ClassVar[NoiseGenerator.Type] = ...
        Hash2D: ClassVar[NoiseGenerator.Type] = ...
        Hash3D: ClassVar[NoiseGenerator.Type] = ...
        Value2D: ClassVar[NoiseGenerator.Type] = ...
        Value3D: ClassVar[NoiseGenerator.Type] = ...
        WarpedValue2D: ClassVar[NoiseGenerator.Type] = ...
        _generate_next_value_: ClassVar[Callable] = ...
        _hashable_values_: ClassVar[list] = ...
        _member_map_: ClassVar[dict] = ...
        _member_names_: ClassVar[list] = ...
        _member_type_: ClassVar[type[object]] = ...
        _unhashable_values_: ClassVar[list] = ...
        _unhashable_values_map_: ClassVar[dict] = ...
        _use_args_: ClassVar[bool] = ...
        _value2member_map_: ClassVar[dict] = ...
        _value_repr_: ClassVar[None] = ...
        __nb_enum__: ClassVar[PyCapsule] = ...
        @classmethod
        def _new_member_(cls, *args, **kwargs): ...
    Gradient2D: ClassVar[NoiseGenerator.Type] = ...
    Gradient3D: ClassVar[NoiseGenerator.Type] = ...
    Hash2D: ClassVar[NoiseGenerator.Type] = ...
    Hash3D: ClassVar[NoiseGenerator.Type] = ...
    Value2D: ClassVar[NoiseGenerator.Type] = ...
    Value3D: ClassVar[NoiseGenerator.Type] = ...
    WarpedValue2D: ClassVar[NoiseGenerator.Type] = ...
    angle: Incomplete
    period: Incomplete
    scale: Incomplete
    seed: Incomplete
    type: Incomplete
    warp_amp: Incomplete
    warp_scale: Incomplete
    x: Incomplete
    y: Incomplete
    z: Incomplete
    def __init__(self) -> None: ...
    def fill(self, arg) -> None: ...
    def generate(self, *args, **kwargs): ...

class Resample:
    _block: int
    _height: int
    _width: int
    angle: float
    input: Incomplete
    map_x: Incomplete
    map_y: Incomplete
    mode: int
    offset_x: float
    offset_y: float
    output: Incomplete
    relative_offset: bool
    sample_mode: int
    scale_by_output_size: bool
    warp_x_strength: float
    warp_y_strength: float
    def __init__(self) -> None: ...
    def process(self) -> None: ...

class Stream:
    def __init__(self) -> None: ...
    def handle(self) -> int: ...
    def sync(self) -> None: ...
    def valid(self) -> bool: ...
