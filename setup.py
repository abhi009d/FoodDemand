from setuptools import setup, find_packages

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements= [req.replace("\n","") for req in requirements]
setup(
    name='Food Demand',
    version='0.0.1',
    author='Abhishek',
    author_email='abhishek.kpv040@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
    ]
)