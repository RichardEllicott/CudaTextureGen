import enum
import nanobind
from _typeshed import Incomplete
from typing import Callable, ClassVar

blur: nanobind.nb_func
cuda_hello: nanobind.nb_func
generate_ao_map: nanobind.nb_func
generate_normal_map: nanobind.nb_func
print_debug_info: nanobind.nb_func
test123: nanobind.nb_func

class DeviceArrayFloat1D:
    array: Incomplete
    def __init__(self) -> None: ...
    def dev_ptr(self) -> int: ...
    def size(self) -> int: ...
    @property
    def shape(self) -> list[int]: ...

class DeviceArrayFloat2D:
    array: Incomplete
    def __init__(self) -> None: ...
    def dev_ptr(self) -> int: ...
    def size(self) -> int: ...
    @property
    def shape(self) -> list[int]: ...

class DeviceArrayFloat3D:
    array: Incomplete
    def __init__(self) -> None: ...
    def dev_ptr(self) -> int: ...
    def size(self) -> int: ...
    @property
    def shape(self) -> list[int]: ...

class DeviceArrayInt1D:
    array: Incomplete
    def __init__(self) -> None: ...
    def dev_ptr(self) -> int: ...
    def size(self) -> int: ...
    @property
    def shape(self) -> list[int]: ...

class DeviceArrayInt2D:
    array: Incomplete
    def __init__(self) -> None: ...
    def dev_ptr(self) -> int: ...
    def size(self) -> int: ...
    @property
    def shape(self) -> list[int]: ...

class DeviceArrayInt3D:
    array: Incomplete
    def __init__(self) -> None: ...
    def dev_ptr(self) -> int: ...
    def size(self) -> int: ...
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

class FluidSimulation:
    _block: Incomplete
    cell_size: Incomplete
    damping: Incomplete
    dt: Incomplete
    gravity: Incomplete
    height: Incomplete
    height_map: Incomplete
    mode: Incomplete
    steps: Incomplete
    water_map: Incomplete
    water_map_next: Incomplete
    water_map_previous: Incomplete
    wave_speed: Incomplete
    width: Incomplete
    wrap: Incomplete
    def __init__(self) -> None: ...
    def process(self) -> None: ...

class GraphNode:
    output: Incomplete
    def __init__(self) -> None: ...
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

class Tectonics:
    _block: Incomplete
    height: Incomplete
    height_map: Incomplete
    test: Incomplete
    width: Incomplete
    def __init__(self) -> None: ...
    def process(self, *args, **kwargs): ...

class TemplateClass4:
    class Type(enum.Enum):
        __new__: ClassVar[Callable] = ...
        APPLE: ClassVar[TemplateClass4.Type] = ...
        ORANGE: ClassVar[TemplateClass4.Type] = ...
        POTATO: ClassVar[TemplateClass4.Type] = ...
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
    APPLE: ClassVar[TemplateClass4.Type] = ...
    ORANGE: ClassVar[TemplateClass4.Type] = ...
    POTATO: ClassVar[TemplateClass4.Type] = ...
    _block: Incomplete
    _height: Incomplete
    _width: Incomplete
    device_array_2d_test: Incomplete
    height_map: Incomplete
    image: Incomplete
    test_bool: Incomplete
    test_float: Incomplete
    test_int: Incomplete
    def __init__(self) -> None: ...
    def process(self, *args, **kwargs): ...

class TemplateDArray1:
    _block: int
    _height: int
    _width: int
    device_array_3d: Incomplete
    device_array_n2d_test: Incomplete
    device_array_n3d_test: Incomplete
    height_map_out: Incomplete
    image: Incomplete
    test_bool: bool
    test_float: float
    test_int: int
    def __init__(self) -> None: ...
    def process(self) -> None: ...
    def test_process(self) -> None: ...
    def test_process2(self) -> None: ...

class TemplateDTest:
    _block: int
    _height: int
    _width: int
    device_array_n2d_test: Incomplete
    device_array_n3d_test: Incomplete
    image: Incomplete
    test_bool: bool
    test_float: float
    test_int: int
    def __init__(self) -> None: ...
    def process(self) -> None: ...
