from glob import glob

from setuptools import find_packages, setup

package_name = 'trinity_missions'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wego',
    maintainer_email='wego@todo.todo',
    description='Mission nodes (slow, crosswalk, cone, ...) driven by the master node',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'slow_node = trinity_missions.slow_node:main',
            'crosswalk_node = trinity_missions.crosswalk_node:main',
            'cone_node = trinity_missions.cone_node:main',
            'three_road_node = trinity_missions.three_road_node:main',
            'parking_node = trinity_missions.parking_node:main',
        ],
    },
)
