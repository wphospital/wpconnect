from setuptools import setup

def readme():
    """Returns the README page"""
    with open('README.rst') as f:
        return f.read()

"""setup lists the configurations to set up and connect to wpconnect"""
setup(
    
    name='wpconnect',
    version='2.0',
    description='Internal package for convenience functions in data warehouse',
    long_description=readme(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
    url='https://github.com/wphospital/wpconnect',
    author='WPH DNA',
    author_email='WPHDataAnalytics@wphospital.org',
    license='MIT',
    packages=['wpconnect'],
    install_requires=[
        'markdown',
        'pyodbc',
        'pandas',
        'cx_Oracle',
        'sqlalchemy',
        'PyGithub',
        'psycopg2',
        'pyyaml',
        'plotly',
        'scipy'
    ],
    package_data={'wpconnect': ['oracle_dlls/*.dll', 'queries/*.sql', 'rpm-queries/*.sql', 'rpm-cfg.yml']},
    include_package_data=True,
    zip_safe=False
)
