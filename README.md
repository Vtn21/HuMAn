<!-- <p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://i.imgur.com/6wj0hh6.jpg" alt="Project logo"></a>
</p> -->

<h1 align="center">HuMAn: Human Motion Anticipation</h1>

<div align="center">

  [![GitHub issues](https://img.shields.io/github/issues/Vtn21/HuMAn)](https://github.com/Vtn21/HuMAn/issues)
  ![GitHub pull requests](https://img.shields.io/github/issues-pr/Vtn21/HuMAn)
  [![GitHub forks](https://img.shields.io/github/forks/Vtn21/HuMAn)](https://github.com/Vtn21/HuMAn/network)
  [![GitHub stars](https://img.shields.io/github/stars/Vtn21/HuMAn)](https://github.com/Vtn21/HuMAn/stargazers)
  [![GitHub license](https://img.shields.io/github/license/Vtn21/HuMAn)](https://github.com/Vtn21/HuMAn/blob/main/LICENSE)

</div>

---

<p align="center"> An AI human motion prediction algorithm
    <br>
</p>

## 📝 Table of Contents
- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Author](#author)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>
The main inspiration for developing this algorithm is *exoskeleton transparency control*, which aims at achieving synchronization and synergy between the motions of the exoskeleton robot and the human user. By being able to predict future motions from a previous time sequence, HuMAn can provide anticipation to a chosen control strategy.

## 🏁 Getting Started <a name = "getting_started"></a>
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
<!-- See [deployment](#deployment) for notes on how to deploy the project on a live system. -->

### 🛠 Prerequisites
What things you need to install the software and how to install them.

This algorithm is programmed using [Python](https://www.python.org/), currently using version 3.8. Installing Python through [Anaconda](https://www.anaconda.com/products/individual) 🐍 is recommended because:

- You gain access to [Conda](https://anaconda.org/anaconda/repo) packages, apart from [Pip](https://pypi.org/) packages;
- Conda is a great tool for managing virtual environments (you can [create](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) one to install all the prerequisites for HuMAn)!

Other key dependencies are (version numbers are kept for reference, but newer versions may work):

- [TensorFlow](https://www.tensorflow.org/) (version 2.4)
```bash
pip install tensorflow
```

- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-zone) (version 11.0)
  - This is not mandatory, but highly recommended! An available NVIDIA GPU can speed up TensorFlow code to a great extent, when compared to running solely on CPU;
  - You can install it with Conda, enabling different versions of the toolkit to be installed in other virtual environments;
  - Ensure to pair TensorFlow and CUDA versions correctly (see [this](https://www.tensorflow.org/install/gpu#software_requirements)).
```bash
conda install cudatoolkit
```

- [STAR model](https://github.com/Vtn21/STAR) (more about it below)
  - The authors of the STAR body model provide loaders based upon Chumpy, PyTorch and TensorFlow. I created a fork of their repository, to make pointing to the model (.npz files) directory easier and more flexible. You can install it using pip:

```bash
pip install git+https://github.com/Vtn21/STAR
```

### 🗂 Database and model

HuMAn uses the [AMASS](https://amass.is.tue.mpg.de/) human motion database. Its data is publicly available, requiring only a simple account. The whole database (after uncompressed) has around 23 GB of [NumPy](https://numpy.org/) npz files. Keep it in a directory of your choice.

AMASS data can be visualized using a series of body models, such as [SMPL](https://smpl.is.tue.mpg.de/), [SMPL-H](https://mano.is.tue.mpg.de/) (this comprises hand motions), [SMPL-X](https://smpl-x.is.tue.mpg.de/) (SMPL eXpressive, with facial expressions), or the more recent [STAR](https://star.is.tue.mpg.de/en). HuMAn uses the STAR model as it has fewer parameters than its predecessors, while exhibiting more realistic shape deformations. You can download the models from their webpages, creating an account as done for AMASS.

Please note that the body models are used here just for visualization, and do not interfere in training. Thus, it is easy to incorporate the other models for this purpose.

Update the folder paths in the scripts as required. The example folder structure is given as follows:

    .
    ├── ...
    ├── AMASS
    │   ├── datasets                          # Folder for all AMASS sub-datasets
    |   |   ├── ACCAD                         # A sub-dataset from AMASS
    |   |   |   ├── Female1General_c3d        # Sub-folders for each subject
    |   |   |   |   ├── A1 - Stand_poses.npz  # Each recording is a npz file
    |   |   |   |   └── ...
    |   |   |   └── ...
    |   |   ├── BMLhandball                   # Another sub-dataset (same structure)
    |   |   |   ├── S01_Expert                # Subject sub-folder
    |   |   |   └── ...
    |   |   └── ...
    |   └── models                            # Folder for STAR model (and maybe others)
    |       └── star                          # The downloaded model
    |           ├── female.npz
    |           ├── male.npz
    |           └── neutral.npz
    ├── HuMAn                                 # This repository
    |   └── ...
    └── ...

### 💻 Installing

This (still) is as simple as cloning this repository.

```bash
git clone https://github.com/Vtn21/HuMAn
```

<!-- End with an example of getting some data out of the system or using it for a little demo. -->

<!-- ## 🔧 Running the tests <a name = "tests"></a> -->
<!-- Explain how to run the automated tests for this system. -->

<!-- ### Break down into end to end tests
Explain what these tests test and why

```
Give an example
```

### And coding style tests
Explain what these tests test and why

```
Give an example
``` -->

## 🎈 Usage <a name="usage"></a>

More to come...

<!-- ## 🚀 Deployment <a name = "deployment"></a>
Add additional notes about how to deploy this on a live system. -->

<!-- ## ⛏️ Built Using <a name = "built_using"></a>
- [MongoDB](https://www.mongodb.com/) - Database
- [Express](https://expressjs.com/) - Server Framework
- [VueJs](https://vuejs.org/) - Web Framework
- [NodeJs](https://nodejs.org/en/) - Server Environment -->

## ✍️ Author <a name = "author"></a>

<a href="https://github.com/Vtn21">
 <img style="border-radius: 50%;" src="https://avatars.githubusercontent.com/u/13922299?s=460&u=2e2554bb02cc92028e5cba651b04459afd3c84fd&v=4" width="100px;" alt=""/>
 <br />
 <sub><b>Victor T. N. 🤖</b></sub></a>

Made with ❤️ by [@Vtn21](https://github.com/Vtn21)

<!-- [![Gmail Badge](https://img.shields.io/badge/-victor.noppeney@usp.br-c14438?style=flat-square&logo=Gmail&logoColor=white&link=mailto:victor.noppeney@usp.br)](mailto:victor.noppeney@usp.br) -->

<!-- -  - Idea & Initial work -->

<!-- See also the list of [contributors](https://github.com/kylelobo/The-Documentation-Compendium/contributors) who participated in this project. -->

## 🎉 Acknowledgements <a name = "acknowledgement"></a>
- [AMASS](https://github.com/nghorbani/amass) by [Nima Ghorbani](http://nghorbani.github.io/)
- [STAR](https://github.com/ahmedosman/STAR) by [Ahmed A. A. Osman](https://ps.is.mpg.de/person/aosman)
- [SPL](https://github.com/eth-ait/spl) by [Emre Aksan](https://ait.ethz.ch/people/eaksan/) and [Manuel Kaufmann](https://ait.ethz.ch/people/kamanuel/)
