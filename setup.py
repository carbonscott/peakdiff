import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="peakdiff",
    version="23.12.05",
    author="Cong Wang",
    author_email="wangimagine@gmail.com",
    description="A peak difference viewer.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carbonscott/peakdiff",
    keywords = ['peakdiff'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points = {
        'console_scripts' : [
            'peakdiff-visualizer-cxi=peakdiff.cxi.serve:main',
            'peakdiff-visualizer-stream=peakdiff.crystfel_stream.serve:main',
        ],
    },
    python_requires='>=3.6',
    include_package_data=True,
)
