from setuptools import find_packages, setup

package_name = "pinkrab_core"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Woolim Park",
    maintainer_email="woolimi91@naver.com",
    description="What a Pinkrab is, and how to talk to its servos.",
    license="Apache-2.0",
)
