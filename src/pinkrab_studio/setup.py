from pathlib import Path

from setuptools import find_packages, setup

package_name = "pinkrab_studio"
static = [str(p) for p in Path(package_name, "static").iterdir() if p.is_file()]

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    package_data={package_name: ["static/*"]},
    include_package_data=True,
    install_requires=["setuptools", "pinkrab_core", "pinkrab_mujoco", "pinkrab_umi",
                      "pinkrab_policy", "hidapi"],
    zip_safe=False,
    maintainer="Woolim Park",
    maintainer_email="woolimi91@naver.com",
    description="Browser view of the Pinkrab robot.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "pinkrab-studio = pinkrab_studio.cli:main",
            "pinkrab-footswitch = pinkrab_studio.footswitch:main",
            "pinkrab-remote = pinkrab_studio.remote_cli:main",
        ],
    },
)
