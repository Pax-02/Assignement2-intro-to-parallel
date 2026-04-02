# Adaptive Simpson Integration with MPI

## Overview

This project solves definite integrals on the interval \([0,1]\) using Adaptive Simpson’s Rule in three modes:

- **Mode 0**: Serial baseline
- **Mode 1**: Static MPI decomposition
- **Mode 2**: Dynamic master/worker MPI decomposition

The program supports these function IDs:

- **0**: `sin(x) + 0.5*cos(3x)`
- **1**: `1 / (1 + 100(x - 0.3)^2)`
- **2**: `sin(200x) * exp(-x)`

---

## Compile Command

```bash
mpicc integration.c -lm -o integration
```

## Run the Program

```bash
mpirun -np P ./integration func_id mode tol
```

### Parameters

    P = number of MPI processes
    func_id = function ID (0, 1, or 2)
    mode = execution mode (0, 1, or 2)
    tol = error tolerance (for example 1e-6 or 1e-8)

### Run commands example

#### Mode 0 (serial Baseiline)

    mpirun -np 1 ./integration 0 0 1e-6
    mpirun -np 1 ./integration 1 0 1e-8
    mpirun -np 1 ./integration 2 0 1e-6

#### Mode 1 (Static MPI)

    mpirun -np 4 ./integration 0 1 1e-6
    mpirun -np 4 ./integration 1 1 1e-8
    mpirun -np 4 ./integration 2 1 1e-6

#### Mode 2 Dynamic MPI

    mpirun -np 4 ./integration 0 2 1e-6
    mpirun -np 4 ./integration 1 2 1e-8
    mpirun -np 4 ./integration 2 2 1e-6

## Machine Specification

    Machine: MacBook Pro (16-inch, 2019)
    CPU: 2.3 GHz 8-Core Intel Core i9
    RAM: 16 GB 2667 MHz DDR4
    Operating System: macOS

## MPI Version

    mpirun (Open MPI) 5.0.9

## Authorship Statement

I Ishimwe Pacis Hanyurwimfura confirm that :

- All code in this repository was written and understood by me
- I implemented each mode (0–2) incrementally, and used git commits to track my progress
- Any external resources I consulted were for general MPI and C language reference, and no code was directly copied from other students or online solutions.

## LINK to Github

    https://github.com/Pax-02/Assignement2-intro-to-parallel.git
