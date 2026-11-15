from glob import glob

from setuptools import find_packages, setup

package_name = 'yolopx_line'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test', 'tools', 'tools.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/models', glob('models/*.engine')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wego',
    maintainer_email='wego@todo.todo',
    description='YOLOPX lane / drivable-area segmentation with TensorRT',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolopx_node = yolopx_line.yolopx_node:main',
        ],
    },
)
