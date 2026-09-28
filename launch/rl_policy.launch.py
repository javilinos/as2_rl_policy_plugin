"""The motion controller node with the rl_policy plugin."""

import os
import re
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
import yaml

PARAMETERS = 'ros__parameters'


class _Loader(yaml.SafeLoader):
    """YAML as rcl types it: 1e-7 is a double, not a string."""


_Loader.add_implicit_resolver(
    'tag:yaml.org,2002:float',
    re.compile(r'''^(?:[-+]?[0-9][0-9_]*\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+
    |[-+]?\.[0-9_]+(?:[eE][-+]?[0-9]+)?
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$''', re.X),
    list('-+0123456789.'))


def merged(base: dict, update: dict) -> dict:
    """Base with update written over it, nested mappings key by key."""
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged(base[key], value)
        else:
            base[key] = value
    return base


def node_blocks(document: dict, name: str, blocks: dict) -> bool:
    """Collect each node's parameters under its full name; True when a block also nests nodes."""
    mixed = False
    for key, value in document.items():
        if key == PARAMETERS:
            merged(blocks.setdefault(name, {}), value or {})
            mixed = mixed or len(document) > 1
        elif isinstance(value, dict):
            mixed = node_blocks(value, f'{name}/{key}' if name else key, blocks) or mixed
    return mixed


def rcl_readable(path: str) -> str:
    """The file, or a copy with one block per node where it nests nodes beside parameters."""
    with open(path, 'r', encoding='utf-8') as handle:
        document = yaml.load(handle, Loader=_Loader) or {}
    blocks = {}
    if not isinstance(document, dict) or not node_blocks(document, '', blocks):
        return path
    with tempfile.NamedTemporaryFile(
            'w', prefix='rl_policy_', suffix='.yaml', delete=False) as handle:
        yaml.safe_dump(
            {name: {PARAMETERS: parameters} for name, parameters in blocks.items()}, handle,
            sort_keys=False)
    return handle.name


def given_files(context) -> list:
    """The parameter files given, plugin_config_file before config_file so the latter wins."""
    files = []
    for name in ('plugin_config_file', 'config_file'):
        path = LaunchConfiguration(name).perform(context)
        if not path:
            continue
        if not os.path.isfile(path):
            raise RuntimeError(f'{name} not found: {path}')
        files.append(rcl_readable(os.path.abspath(path)))
    if not files:
        raise RuntimeError(
            'rl_policy needs config_file, plugin_config_file or both: they name the policies')
    return files


def controller_node(context) -> list:
    """The controller manager with every parameter layer, in precedence order."""
    as2_config = os.path.join(
        get_package_share_directory('as2_motion_controller'),
        'config', 'motion_controller_default.yaml')
    package_folder = get_package_share_directory('as2_rl_policy')
    plugin_defaults = os.path.join(package_folder, 'config', 'rl_policy_default.yaml')
    available_modes = os.path.join(package_folder, 'config', 'available_modes.yaml')
    return [
        Node(
            package='as2_motion_controller',
            executable='as2_motion_controller_node',
            name='controller_manager',
            namespace=LaunchConfiguration('namespace'),
            output='screen',
            emulate_tty=True,
            arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
            parameters=[
                as2_config,
                plugin_defaults,
                *given_files(context),
                # Last, so no file can load another plugin or turn the bypass on.
                {
                    'plugin_name': 'rl_policy',
                    'use_bypass': False,
                    'plugin_available_modes_config_file': available_modes,
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                },
            ],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    """Return the launch description."""
    return LaunchDescription([
        DeclareLaunchArgument(
            'namespace', description='Drone namespace',
            default_value=EnvironmentVariable('AEROSTACK2_SIMULATION_DRONE_ID',
                                              default_value='drone0')),
        DeclareLaunchArgument(
            'use_sim_time', description='Use the simulation clock', default_value='false'),
        DeclareLaunchArgument(
            'plugin_config_file', default_value='',
            description='Plugin parameters (the rl_policy block), over the package defaults'),
        DeclareLaunchArgument(
            'config_file', default_value='',
            description='Node parameters (a controller_manager block) or a whole flight, '
                        'over plugin_config_file'),
        DeclareLaunchArgument('log_level', description='Logging level', default_value='info'),
        OpaqueFunction(function=controller_node),
    ])
