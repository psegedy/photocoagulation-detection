from setuptools import setup

setup(
    name="laser-marks",
    version="0.1",
    py_modules=["laser_marks"],
    install_requires=["Click", "opencv-python", "numpy"],
    entry_points="""
        [console_scripts]
        laser-marks=laser_marks:cli
    """,
)
