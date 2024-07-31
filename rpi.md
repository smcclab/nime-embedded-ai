---
layout: page
title: Raspberry Pi and IMPSY
permalink: /rpi/
---

# Introduction

This workshop is a tutorial in running the IMPSY musical machine learning system on Raspberry Pi single board computers. The idea is to use this technology to create new intelligent musical instruments.

From a broader perspective, this workshop is a deep dive in how to get Python-based AI/ML software working well on a Raspberry Pi and how it can interact with typical music software such as Pure Data or hardware devices such as a MIDI USB synthesiser

## Background

This workshop focuses on two technical areas: Python AI software development and Raspberry Pi programming. Both areas have lots of details and recommended practices might be different for different projects. The assumption here is that the software (IMPSY) is going to be deployed on a Rapsberry Pi that is more-or-less permanently integrated into a digital musical instrument. So, we will do things a bit differently that you might for a Raspberry Pi server, media player, etc.

An assumption about the workshop is that you might know a bit about Python (e.g., from coding classes or your own experiments) and a bit about using a Raspberry Pi (e.g., you might have tried operating one from a text command line).

This tutorial will work with: Raspberry Pi 3 Model B+, Raspberry Pi 4, Raspberry Pi 5, Raspberry Pi Zero 2 W. These are the models that specifically support the 64-bit Raspberry Pi OS which is required. I have been using RPi Zero 2 W because they are the _smallest_ and _cheapest_ and thus best[^1] for creating many musical instruments. The RPi 5 is _much_ faster and better in almost every way as long as you are ok with spending more money and having a physically larger board. If you are bringing your own Pi, make it a 4 or 5! Life will be good.

## Requirements

- Laptop with MacOS, Linux, or Windows which has: Docker and Python 3.11+
- Optional: Raspberry Pi 4, 5, Zero 2 W (I will bring Raspberry Pi Zero 2 W's for testing if you want to borrow one).
- A sense of adventure.

N.B.: if you are bringing a Raspberry Pi, please bring a **microSD card** (16GB or greater) that you _are comfortable erasing_.

## Checkpoints

By the end of the workshop, you should have tried out doing some of the following things

- [ ] Run IMPSY on your computer using Docker (or installing with Poetry)
- [ ] Collect data, train models, and run IMPSY on your computer
- [ ] Run IMPSY on a Raspberry Pi
- [ ] Transfer a trained model to a Raspberry Pi and perform with it
- [ ] Retrieve more data from the Raspberry Pi

# Workshop Tasks

## 0. Learning about what IMPSy is, and how it works.

IMPSY is a generative AI system designed to integrate into typical NIMEs. IMPSY's job is to receive inputs from a NIME's interface (e.g., knobs, sensors, controllers, etc) and send outputs to a NIME's sound/music generators (e.g., a MIDI synthesiser, Pd patch, Processing sketch).

![IMPSY sits in between a musician's control interface and the sound output.]({% link assets/impsy/predictive_interaction.png %})

The machine learning model in IMPSY is trained to produce the _next_ action in a sequence of control signals. A unique aspect of IMPSY's model is that it can work with a number of control signals simultaneously and that it also predicts the _time_ that the next action will occur. This means that it is particularly useful in NIMEs where we might have a number of sensors (e.g., 8 knobs, 12 light sensors) working together, and where we might be performing music with free or irregular rhythm.

IMPSY is usually configured to take over control of a NIME _when the musician is not playing_, but allowing signals from the musician to go straight through to the sound generator. 
It's also possible to use it as a filter between the musician's actions and and the parameters sent to a sound generator. 
Other configurations might be possible and exploring these is one reason to work with this system!

The machine learning model in IMPSY is a recurrent neural network using long short-term memory (LSTM) cells and a mixture density distribution for determining outputs. This is, now, a fairly well-explored design and one that has [been previously shown](http://www.nime.org/proceedings/2019/nime2019_paper050.pdf) to work well in NIME designs on devices as inexpensive as a Raspberry Pi.

![IMPSY's MDN design]({% link assets/impsy/mdn_diagram.png %})

An LSTM-RNN is a typical design for AI systems that learn to [generate sequences](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). Using a mixture density network (MDN) at the end (as IMPSY does) allows us to generate multi-dimensional and continuous valued data and gives us some high-level controls over how "random" the generated choices will be. As [Leitman (2017)](http://doi.org/10.5281/zenodo.1176197) tells us, continuous sensors make for more effective NIMEs and so this mixture density recurrent neural network (MDRNN) is a good choice for NIME interaction.

Learning all the details of how to create MDRNNs is a bit beyond this workshop, but here are some slide decks (presented at a _previous_ NIME workshop about these systems):

- [Deep Dive on RNNs](https://creativeprediction.xyz/presentations/deep-dive-on-rnns/)
- [Presentation on Mixture Density Networks](https://creativeprediction.xyz/presentations/mixture-density-networks/)
- [Making Predictive NIMEs with Neural Networks](https://creativeprediction.xyz/presentations/intro/)
- [An Interactive Musical Prediction System with MDRNNs](https://creativeprediction.xyz/presentations/imps)

While LSTM-RNNs are a bit out of fashion compared with the _transformer_ approaches used in present large language models (LLMs), the purpose of this workshop is efficient neural network designs that work in an embedded setting (no data centres and no GPUs required), so MDRNNs are still a very good choice in this context.

**TODO** Some more on MIDI and OSC and how IMPSY might need to be configured.

## 1. Installing IMPSY on your computer

IMPSY is distributed as a code repository at <https://github.com/cpmpercussion/impsy>. We're going to try installing it in two ways: as a docker image and natively using Python and Poetry.

The downside of the docker container is that it won't know about MIDI devices on your system (OSC communication is fine).

The downside of the Python/Poetry method is that configuring your system with Python can be an [absolute nightmare](https://xkcd.com/1987/).

Start with Docker and try out Python afterwards if you can get it working, it's important to be ok with abandoning the Python/Poetry install if it's beyond you as it might take too long and we have other things to do in this workshop!

Here's what to do:

1. download IMPSY, either cloning the repo above or, at the website, click "Code" (big green button) and download zip.

2. **Docker Install**: ensure that you have _docker_ on your computer, if you don't, head to <https://www.docker.com> to get an installer and install it.

  - using a command line, navigate to the directory where IMPSY's code is (e.g., for me on macos it's `/Users/charles/src/impsy`)

  - **Install**: run `docker pull charlepm/impsy` -- this will download a docker image which has `impsy` (and it's dependencies) preinstalled

  - **Run the container**: you can try out the docker container by running: `docker run -it charlepm/impsy`, this should leave you in a bash prompt with something like `root@07a13e52f703:/impsy#`

  - **Try some impsy commands**: you can see the commands in IMPSY by typing `poetry run ./start_impsy.py --help` into the bash prompt of your docker container. 
  
  - You might like to try the command `poetry run ./start_impsy.py test-mdrnn` which sets up IMPSY's MDRNN to check everything is working. 

  - Try the command `poetry run ./start_impsy.py run` to run IMPSY's interaction server. This should start generating data using the default configuration: a pre-trained 8-channel MDRNN, you'll see the outputs shown on the screen in MIDI format (numbers between 0 and 127).

  - You can type `Ctrl-C` (control key and then c held together) to exit IMPSY and type `exit` to exit the Docker container.

3. **Docker Compose Setup**: let's try a whole IMPSY setup. This will involve starting two docker containers with the same image, one for IMPSY's `run` command (as above) and one for a web server that we can use to control IMPSY while it's running. Look in the `docker-compose.yml` file to see how this is set up.

  - **Lookign in `docker-compose.yml`: this file sets up two `charlepm/impsy` containers, and maps the important folders from the `impsy` directory (on your computer) to their location in the containers. The folders are `datasets`, `logs`, `models`, and one file `config.toml`. The 

  - run the command `docker-compose -f docker-compose.yml up` to bring everything online (yes its that easy, if you haven't tried docker compose before it feels a bit magical)

  - you can see that it's working by going to `http://127.0.0.1:4000` in a browser

  - **why are we looking in a browser now?** IMPSY has a web interface that lets you see/download/edit: log files that have been collected, datasets that have been collected, available model files, and the main configuration file for IMPSY `config.toml`.

  - **editing `config.toml`**: Now that IMPSY is up and running let's edit config.toml and define an interaction with a basic Pure Data patch. Remember that IMPSY in a Docker container can't use MIDI, but we can interact using OSC with Pure Data.

4. **Basic IMPSY setup with Pure Data**.

5. **Set up IMPSY with Python**: for this you will need: _python 3.11_ and `poetry` to be installed on your computer. IMPSY really needs Python 3.11.x to work (not 3.8, not 3.12... etc), so you might need to think about having multiple pythons available.

  - run `python --version` at a command line to see what version of Python you have.

  - if you need to install **Python 3.11** you can theoretically get Python from [here](https://www.python.org/downloads/release/python-3119/), but you might have better luck installing `pyenv` to manage different python versions, you can see it [here](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation).

  - **Poetry**: Poetry is a dependency manager for Python packages, see <https://python-poetry.org>. To install Poetry, run this command: `curl -sSL https://install.python-poetry.org | python3 -`

  - You can test your poetry install by running `poetry` on a command line.

  - **Install IMPSY**: now you can run `poetry install` to install IMPSY's dependencies (hope this works!).

  - **Test out IMPSY**: you should be able to test out IMPSY by running the same commands as inside the docker container: e.g., `poetry run ./start_impsy.py test-mdrnn`.


## 2. Collecting some data and training a model

## 3. Running IMPSY on a Raspberry Pi


## 4. Transferring a trained model to the Raspberry Pi


## 5. Retrieving more data from the Raspberry Pi


# Wrap up



[^1]: _arguably_ the best I suppose 🙃

