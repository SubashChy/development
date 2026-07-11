from setuptools import find_packages , setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirement(file_path:str) -> List[str]:
    requirement =[]
    with open(file_path) as file_obj:
       requirement = file_obj.readlines()
       requirement = [req.replace('\n' , "") for req in requirement] 
       
       if HYPHEN_E_DOT in requirement:
           requirement.remove(HYPHEN_E_DOT)
       

setup(
    name='ml project',
    version='0.0.1',
    author='subash chaudhary',
    author_email='taylorlaunter0000@gmail.com',
    packages= find_packages(),
    install_requires= get_requirement('requirements.txt')
    
)