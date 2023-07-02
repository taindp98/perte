import setuptools,re,sys

setuptools.setup(
    name = 'perte',         
    packages = ['perte'],
    version = '0.1',     
    license='MIT', 
    description = 'A fast way to use the diverse loss functions in deep learning', 
    author = 'Rémi NGUYEN',                   
    author_email = 'taindp98@gmail.com',      
    url = 'https://github.com/taindp98/perte',   
    download_url = 'https://github.com/taindp98/perte/archive/refs/tags/v0.1.tar.gz',   
    keywords = ['loss-function', 'discrimination'],  
    install_requires=[
        'pip', 'packaging',
        'torch', 'numpy'
    ],
    classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    ],
)