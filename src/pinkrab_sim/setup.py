from setuptools import find_packages, setup

package_name = "pinkrab_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "pinkrab_core", "pinkrab_mujoco", "pinkrab_umi", "pinkrab_policy"],
    zip_safe=False,
    maintainer="Woolim Park",
    maintainer_email="woolimi91@naver.com",
    description="Pinkrab in simulation: Hermit takes, and a policy judged on the simulated arm.",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "pinkrab-sim = pinkrab_sim.cli:main",
    ]},
)
