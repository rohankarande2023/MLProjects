from setuptools import setup,find_packages
from typing import List

def get_requirements(requirements_path:str)->List[str]:
    """This function will return a list of requirements"""
    requirements=[]
    dot_e='-e .'
    with open(requirements_path) as f:
        requirements=f.readlines()
        requirements=[x.replace('\n', "") for x in requirements]

        if dot_e in requirements:
            requirements.remove(dot_e)
    return requirements



setup(

    name='mlprojects',
    version='0.0.1',
    author='Rohan',
    author_email='rohankarande83@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')


)