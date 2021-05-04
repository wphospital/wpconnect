from setuptools import setup

# setup(
#     name='wpconnect',
#     url='https://github.com/wphospital/wpconnect',
#     author='Jon Sege',
#     author_email='jsege@wphospital.org',
#     packages=['wpconnect'],
#     install_requires=['markdown', 'pyodbc', 'pandas', 'cx_Oracle'],
#     version='0.1',
#     license='MIT',
#     description='Internal package for convenience functions in data warehouse',
#     include_package_data=True,
# )

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='wpconnect',
    version='0.1',
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
        'cx_Oracle'
    ],
    package_data={'wpconnect': ['*.dll', 'queries/*.sql']},
    include_package_data=True,
    zip_safe=False
)
