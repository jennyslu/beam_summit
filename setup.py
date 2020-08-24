import setuptools

# NOTE: Any additional file besides the `main.py` file has to be in a module
#       (inside a directory) so it can be packaged and staged correctly for
#       cloud runs.

REQUIRED_PACKAGES = [
    'apache-beam[gcp]==2.16.*',
    'numpy==1.17.*',
    'pandas==0.25.*',
    'tensorflow-transform==0.15.*',
    'tensorflow==2.0.*',
]

setuptools.setup(
    name='iowa_sales',
    version='0.0.1',
    author='Jenny Lu',
    author_email='jenny.lu@pltalot.com',
    description='Apache Beam Summit workshop',
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
)
