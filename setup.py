from setuptools import find_packages, setup
from typing import List

def get_requirements()->List[str]:
    """
    Returns a list of requirements
    """
    requirement_lst:List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            lines = file.readlines()
            for line in lines:
                requirement= line.strip()

                if requirement and requirement!= '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_lst

setup(
    name="PaySim-Fraud-Detection",
    version="0.0.1",
    author="Aditya Ranganekar",
    author_email="ranganekaraditya@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)