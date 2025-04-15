from setuptools import setup, find_packages

# Leer las dependencias del archivo requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="my_relativity",
    version="0.1.0",
    author="Luis Daniel Díaz",
    author_email="lldddv@gmail.com",
    description="Este paquete proporciona funciones de apoyo para crear gráficas y simulaciones básicas de relatividad.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/LuisDanieDiaz/RelativityGravitationAndCosmology",
    packages=find_packages(),
    install_requires=requirements,  # Incluir las dependencias
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    include_package_data=True,
    package_data={
        "my_relativity": ["plots/images/**/*.png"],
    },
)