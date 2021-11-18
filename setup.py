from setuptools import setup

setup(
    name='libratools',
    version='0.0.1',
    packages=['libratools'],
    install_requires=[
        'requests',
        'importlib; python_version == "3.7.7"',
    ],
    # metadata to display on PyPI
    author = "Vincent (Vince) J. Straub",
    author_email = "vincejstraub@gmail.com",
#     description = "Tools for pre-processing high-throughput animal tracking data.",
#     keywords = "tool, dashboard, tracking",
    # url = "http://example.com/HelloWorld/",   # project home page, if any
    # project_urls = {
    #     "Bug Tracker": "https://bugs.example.com/HelloWorld/",
    #     "Documentation": "https://docs.example.com/HelloWorld/",
    #     "Source Code": "https://code.example.com/HelloWorld/",
    # },
    zip_safe = False,
)
