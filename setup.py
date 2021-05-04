from setuptools import setup, find_packages

setup(
    name='wpconnect',
    url='https://github.com/wphospital/wpconnect',
    author='Jon Sege',
    author_email='jsege@wphospital.org',
    packages=['wpconnect'],
    install_requires=['markdown', 'pyodbc', 'pandas', 'cx_Oracle'],
    version='0.1',
    license='MIT',
    description='Internal package for convenience functions in data warehouse',
    include_package_data=True,
)
