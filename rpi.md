---
layout: page
title: Raspberry Pi and IMPSY
permalink: /rpi/
---

# Introduction

This workshop is a tutorial in running the IMPSY musical machine learning system on Raspberry Pi single board computers. The idea is to use this technology to create new intelligent musical instruments.

From a broader perspective, this workshop is a deep dive in how to get Python-based AI/ML software working well on a Raspberry Pi and how it can interact with typical music software such as Pure Data or hardware devices such as a MIDI USB synthesiser

## Background

This workshop two technical areas: Python AI software development and Raspberry Pi programming. Both areas have lots of details and recommended practices might be different for different projects. The assumption here is that the software (IMPSY) is going to be deployed on a Rapsberry Pi that is more-or-less permanently integrated into a digital musical instrument. So, we will do things a bit differently that you might for a Raspberry Pi server, media player, etc.

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

## 1. Installing IMPSY on your computer

IMPSY is distributed as a code repository at <https://github.com/cpmpercussion/imps>




## 2. Collecting some data and training a model

## 3. Running IMPSY on a Raspberry Pi


## 4. Transferring a trained model to the Raspberry Pi


## 5. Retrieving more data from the Raspberry Pi


# Wrap up



[^1]: _arguably_ the best I suppose 🙃

