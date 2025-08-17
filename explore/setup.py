from setuptools import setup

package_name = 'explore'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    py_modules=[],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'opencv-python'],
    zip_safe=True,
    maintainer='kruthik',
    maintainer_email='kruthik@example.com',
    description='Explore package',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'twist_to_stamp = explore.twist_to_stamp:main'
        ],},
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_detector_node = aruco_detector_node.aruco_detector_node:main'
        ],
    },
    # entry_points={
    # 'console_scripts': [
    #     'marker_detector_node = marker_detector.marker_detector_node:main',
    #     ],
    # },
)
