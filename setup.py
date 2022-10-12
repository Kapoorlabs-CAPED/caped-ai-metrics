from setuptools import find_packages, setup
from os import path 

_dir = path.dirname(__file__)

with open('README.md') as f:
    long_description = f.read()
with open(path.join(_dir, 'caped_ai_metrics','_version.py'), encoding="utf-8") as f:
    exec(f.read())
    
setup(
    name="caped-ai-metrics",
    version = __version__,
    
    author='Varun Kapoor',
    author_email='randomaccessiblekapoor@gmail.com',
    description='Metric computation toolkit for segmentation and classification tasks',
    long_description = long_description,
    long_description_content_type='text/markdown',
    install_required=[
        'oneat',
        'seaborn'
                      ], 
    
    packages = find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.9',
    ],
    
    
)        
