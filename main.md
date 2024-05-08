# Workshop Title

Building NIMEs with Embedded AI

# Speaker List

## Charles Patrick Martin

Charles Martin is a computer scientist specialising in music technology,
creative AI and human-computer interaction at The Australian National
University, Canberra. Charles develops musical apps such as PhaseRings,
researches creative AI, and performs music with Ensemble Metatone and
Andromeda is Coming. At the ANU, Charles teaches creative computing and
leads research into intelligent musical instruments. His lab's focus is
on developing new intelligent instruments, performing new music with
them, and bringing them to a broad audience of musicians and performers.
Charles has presented workshops on AI/ML and NIMEs at NIME 2019, 2020,
and 2021 and is an organiser of the Generative AI and HCI workshops at
CHI 2022, 2023, and 2024.

Charles' website is: <https://charlesmartin.au>

## Teresa Pelinski

Teresa Pelinski is a PhD student at the Augmented Instruments Lab at the
Centre for Digital Music, based in Queen Mary University of London.
Teresa's research focuses on developing tools for prototyping with ML in
the context of musical practice, and on doing so from a practice
research lens. Currently, she is also an Enrichment Student at the Alan
Turing Institute. Teresa holds a BSc in Physics from the Universidad
Autónoma de Madrid and a MSc in Sound and Music Computing (MSc) from
Universitat Pompeu Fabra in Barcelona. Teresa was an organiser of the
Embedded AI for NIME 2022 workshop.

Teresa's website is: <https://teresapelinski.com/>

# Workshop Description (750)

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

# Short description (70)

Embedded computing technologies have long been present in NIME but it is
challenging to deploy AI and machine learning models on such systems. In
this hands-on workshop, we will provide participants with starting
points to develop their own NIMEs with embedded platforms and machine
learning models.

# Preferred length of workshop

**Full Day**

# Technical and space requirements

-   Projection screen and sound system for slides and presentations from
    laptop.

-   Class or tutorial room with space for laptops and discussions.

-   Internet connections for all participants (for downloading code):
    preferably a simple ethernet connection to run a basic wireless
    router in the room

-   Power strips for plugging in laptops.

**Presentation Mode:** We are primarily planning for in-person
presentation. A virtual workshop is possible for participants who have
their own Bela or Raspberry Pi at home. We do not suggest running a
hybrid workshop as we would need a dedicated presenter to handle
interactions with online participants and technical support to broadcast
the tutorial content.

# Optional Links

This workshop builds on materials used in previous NIME workshops and
classes at the presenters' institutions, e.g.:

-   Making Predictive NIMEs with Neural Networks:
    <https://creativeprediction.xyz/workshop/>

-   Embedded AI for NIME: <https://embedded-ai-for-nime.github.io/>

-   Critical Perspectives on AI/ML in Musical Interfaces:
    <https://critical-ml-music-interfaces.github.io/>

-   Sound and Music Computing: Generative AI and Computer Music:
    <https://comp.anu.edu.au/courses/laptop-ensemble/lectures/11-genai/>

-   pyBela + pyTorch + Cross-compilation Tutorial: <https://github.com/pelinski/pybela-pytorch-xc-tutorial>

-   IMPSy - the Interactive Musical Prediction SYstem: <https://github.com/cpmpercussion/imps>
[^1]: School of Computing, The Australian National University

[^2]: Centre for Digital Music, Queen Mary University of London
