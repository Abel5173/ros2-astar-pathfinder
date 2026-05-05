import os
from glob import glob
from setuptools import setup

package_name = 'astar_robot_sim'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'),
            glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='abel',
    maintainer_email='abel@todo.todo',
    description='A* path planning simulation',
    license='MIT',
    entry_points={
        'console_scripts': [
            'astar_planner = astar_robot_sim.astar_planner:main',
        ],
    },
)
