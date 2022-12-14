from setuptools import setup, find_packages

setup(
    name="melkor_logic",
    version="1.0",
    author="Yiqi Sun (Zongjing Li)",
    author_email="ysun697@gatech.edu",
    description="Melkor Logic Network.",

    # project main page
    url="http://jiayuanm.com/", 

    # the package that are prerequisites
    packages=find_packages(),
    package_data={
        '':['melkor_logic'],
        'bandwidth_reporter':['melkor_logic']
               },
)

