from setuptools import find_packages, setup

package_name = "pinkrab_teleop"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "pinkrab_core", "pinkrab_umi"],
    zip_safe=False,
    maintainer="Woolim Park",
    maintainer_email="woolimi91@naver.com",
    description="The handheld gripper, driving an arm.",
    license="Apache-2.0",
    entry_points={"console_scripts": ["teleop = pinkrab_teleop.teleop:main"]},
)
