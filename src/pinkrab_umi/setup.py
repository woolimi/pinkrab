from glob import glob

from setuptools import find_packages, setup

package_name = "pinkrab_umi"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/meshes", glob("meshes/*.stl")),
    ],
    package_data={package_name: ["static/*"]},
    include_package_data=True,
    install_requires=["setuptools", "pinkrab_core"],
    zip_safe=False,
    maintainer="Woolim Park",
    maintainer_email="woolimi91@naver.com",
    description="Top-camera 6-DoF tracking of the Pinkrab handheld UMI gripper.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "pinkrab-umi = pinkrab_umi.cli:main",
        ],
    },
)
