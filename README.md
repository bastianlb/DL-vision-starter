# Research Coding Starter Kit

Welcome to my Coding Starter Kit. As you embark on your research journey, mastering the right tools will be paramount to your success. While it's true that some prefer to use raw Pytorch, this guide will introduce you to an assortment of tools and techniques that I have personally found to be tremendously time-saving and beneficial. Remember, these are subjective preferences and should be seen as a point of departure, rather than the final destination. 

Let's start with **PyTorch Lightning**. If you're new to machine learning or deep learning, PyTorch Lightning is a godsend. It's essentially a lightweight PyTorch wrapper that helps you automate and organize your PyTorch code, allowing you to move from research into production with less hassle. It helps you keep your code clean and easy to understand, enabling you to focus on the important parts of your research without getting bogged down by the technicalities.

Next up is **Weights & Biases (wandb)**. Think of wandb as your digital lab notebook. It allows you to keep track of your experiments, log metrics, visualize results, and even share your findings with others. It's incredibly valuable for tracking progress and debugging your models. However, if you prefer, TensorBoard is a viable alternative.

Configuration management is an essential part of any project, especially when you're dealing with complex ML models. Here, **Hydra** comes into play. Hydra allows you to create hierarchical configurations, making your models more flexible and easily adjustable. Alternatives include YACS and Sacred, but what's important is to manage your configurations using files, as this ensures a clearer and more maintainable structure.

Finally, let's talk about **Git**. A version-control system, Git allows you to track your code's history, making it easier to debug and understand how your project has evolved over time. It also facilitates collaboration, enabling multiple people to work on the same codebase without stepping on each other's toes. In a nutshell, Git is like a safety net that preserves your code's history while allowing you to experiment and iterate.

Project and code organization is something that's often overlooked but can make a significant difference in your research. A well-structured project is easier to understand, debug, and maintain. Moreover, it makes your research more reproducible, which is a cornerstone of good scientific practice.

In summary, these tools and best practices will set you up for a smooth and successful research journey. The road ahead is long, and having the right tools at your disposal will make it that much more manageable. Welcome to the exciting world of research!

## Project Outline / Organization

As your project grows, it can be practical to organize python modules in advance. We provide a sample outline that could be used.
As a small demo, this demo includes a lightweight transformer module which is setup to be trained on the popular MNIST training set.
The transformer should achieve >96% accuracy on the validation/test sets, which are identical as a proof of concept here.
The resources required for this are quite low; you could likely train it locally on your own machine. The project outline is as follows:
```
├── README.md
├── DL_kickstart
│   ├── __init__.py
│   ├── conf # configuration files, here we use hydra
│   ├── datasets # the mnist dataset
│   ├── models # conatins the transformer model
│   ├── modules # contains the pytorch lightning module for training
│   └── train.py # main training script
├── requirements.txt # the project requirements
```

## Installation

To get started, create a virtual environment through one of many tools such as `miniconda`, `python virtualenv`, or `poetry`.
After activating your environment, you can install dependencies with for example `pip install -r requirements.txt`.

Then, simple run the main training script. You can override any default configuration arguments in a custom config file,
or as described in the hydra documentation.

Make sure the current directory is in your PYTHONPATH, i.e. `PYTHONPATH=./ python DL_kickstart/train.py`
