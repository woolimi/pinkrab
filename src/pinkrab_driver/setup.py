from setuptools import find_packages, setup

package_name = "pinkrab_driver"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "pinkrab_core"],
    zip_safe=False,
    maintainer="Woolim Park",
    maintainer_email="woolimi91@naver.com",
    description="The servo bus on the ROS graph.",
    license="Apache-2.0",
    entry_points={"console_scripts": ["driver = pinkrab_driver.driver:main"]},
)
