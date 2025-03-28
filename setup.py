from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='wpconnect',
    version='3.2',
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
        'cachelib',
        'markdown',
        'pyodbc',
        'pandas',
        'cx_Oracle',
        'sqlalchemy',
        'neo4j',
        'pandas',
        'PyGithub',
        'psycopg2',
        'pyyaml',
        'plotly',
        'redis',
        'requests',
        'scipy',
        'urllib3<2',
        'weaviate-client',
        'sqlparse',
        'sprucepy @ git+https://github.com/wphospital/sprucepy'
    ],
    package_data={'wpconnect': ['oracle_dlls/*.dll', 'queries/*.sql', 'rpm-queries/*.sql', 'rpm-cfg.yml']},
    include_package_data=True,
    zip_safe=False
)
