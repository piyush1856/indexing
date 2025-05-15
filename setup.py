from setuptools import setup, find_packages


# Read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='indexing',
    version='0.1.0',
    author='Piyush Tyagi',
    author_email='piyushtyagi28@hotmail.com',
    description='indexing',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/piyush1856/indexing',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            # 'command-name=package.module:function',
        ],
    },
)