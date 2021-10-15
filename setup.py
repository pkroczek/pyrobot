from setuptools import find_namespace_packages, setup

packages = [package for package in find_namespace_packages(where='./src', include='pyrobot.*')]

setup(
    name='pyrobot',
    version='1.0.0',
    author='Piotr',
    # author_email='you@yourdomain.com',
    description='Python package for develop manipulator kinematics',
    platforms='Posix;Windows',
    packages=packages,
    package_dir={
        '': 'src'
    },
    include_package_data=True,
    install_requires=(
        'numpy','matplotlib'
    ),
    classifiers=[
        'Development Status :: 1 - Alpha',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)