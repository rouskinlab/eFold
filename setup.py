from setuptools import setup, find_packages

setup(
    name='efold',
    version='0.1.3',
    description='A Python package for RNA folding using end-to-end deep learning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    authors=[
        {'name': 'Yves Martin', 'email': 'yves@martin.yt'},
        {'name': 'Alberic de Lajarte', 'email': 'albericlajarte@gmail.com'},
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.10',
    py_modules=['efold'],
    include_package_data=True,
    package_data={'': ['resources/*.pt']},
    packages=find_packages(),
) 