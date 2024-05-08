---
layout: page
title: About the workshop
permalink: /about/
---

Embedded computing technologies have long been present in NIME.
Single-board computers such as Raspberry Pi or Bela, or microcontrollers
such as Arduino and its variants are at the technical core of many NIMEs
presented every year
[@Ramkissoon2011; @Leeuw2009; @Zayas-Garin2021; @Berdahl2014; @Sullivan2020].
These platforms typically provide APIs that simplify interfacing with
sensors and real-time software implementations, with some platforms
optimised for notably low action-to-sound latency [@McPherson2015].
Furthermore, these platforms allow NIMEs to work off-the-shelf without
relying on a general-purpose laptop which may be prone to compatibility
issues following e.g., operating system updates [@Morreale2017a].

The 'Embedded AI: Challenges and Opportunities' workshop held at NIME
2022 [@Pelinski2022a] exposed some of the challenges related to
integrating deep learning in real-time embedded environments. Some of
these issues related to the low computational capability of these
platforms, such as how to reduce the neural network's size and program
the real-time systems efficiently. However, many other challenges were
related to the lack of established workflows to deploy deep learning
models into such platforms. Given the variety of architectures of these
platforms, practitioners will typically not find instructions for
compiling the libraries on their platforms, and open-source projects
often will not document how the environment in the embedded platform was
set up. Although there are a plethora of educational materials for
learning physical computing (Arduino, Bela, Raspberry Pi) and separately
for AI/ML (Keras, Colab, PyTorch), there is very little overlap to help
beginners getting started with their embedded AI projects in the context
of NIME.

This workshop will introduce participants to the tools and techniques
required to build embedded NIMEs with techniques from machine learning,
generative AI, physical computing, embodied interaction and real-time
artistic performance. We will provide background and code examples for
creating and training neural networks and demonstrate new toolkits for
interactive musical prediction. Building on current research trends, we
aim to provide support for participants who use Bela and Raspberry Pi
for creating NIMEs. Our goal is that participants will engage with the
introduced tools to define new embedded AI NIME prototypes and future
research directions.

This session will be interactive and aims to provide the participants
with some starting points to develop their own embedded AI NIMEs. The
workshop will be structured as follows: introduction to embedded AI and
physical computing (30 min), Raspberry Pi tutorial (3h), Bela tutorial
(3h), and demos and ending (30 min), with multiple breaks as needed. At
the beginning of each tutorial, participants will be given demonstration
code at the beginning of the session that they can run on their laptops
as we go. The workshop materials will be openly available online. We
will accept up to 20 participants and encourage them bring their own a
Bela or Raspberry Pi, although sufficient demonstration kits for
hands-on learning will be provided. Participants will only need to bring
their laptop and headphones. Our goal is that by the end of the
tutorial, each participant will have trained a custom neural network on
their own data, as well as feel confident to create new NIMEs with
machine learning.