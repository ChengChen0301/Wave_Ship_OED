# Variational Goal-Oriented Optimal Experimental Design

Implementation of **Variational Goal-Oriented Optimal Experimental Design for Mixed-Distribution Quantities of Interest: Application to Ship Roll Safety** by Chen Cheng, Xun Huan, and Yulin Pan.

## Overview

This work selects experiments that maximize expected information gain about a quantity of interest (QoI) with both a discrete probability mass and a continuous component. A mixed variational model combines an atom-probability network with a conditional normalizing flow to estimate the Barber–Agakov lower bound. Samples at the atom are scored using probability mass, while continuous samples are scored using density.

## Examples

- **Analytical validation:** A linear-Gaussian observation model with a thresholded QoI, comparing variational estimates against the exact goal-oriented information gain.
- **Ship roll safety:** Surrogate-assisted experimental design for inferring the temporal probability of roll-angle threshold exceedance in random waves. Designs specify wave-group length and amplitude.

## Computational workflow

1. Generate parameter, observation, and QoI samples; for the ship application, construct the roll-response surrogate.
2. Train the mixed variational model at each candidate design.
3. Evaluate the information-gain lower bound on held-out samples.
4. Compare design landscapes and select the design with the largest estimated bound.

See the paper's implementation appendix for network architectures and training settings.

## Citation

Chen Cheng, Xun Huan, and Yulin Pan. *Variational Goal-Oriented Optimal Experimental Design for Mixed-Distribution Quantities of Interest: Application to Ship Roll Safety.*
