# Spinchain_VQE_codes

This repository contained codes written with IBM qiskit for variational quantum eigensolver (VQE) simulation of XXZ spin chains. The parameterized circuits used are based on those constructed with particle-conserving gates, which are composed into a brick-wall pattern, giving the so-called brickwall circuits. The codes use the multiprocess module to parallelize, so as to run multiple jobs simultaneously, e.g. spin chains simulations with varying parameters, such as different lattices sizes, diffrent types of particle-conserving gates, and so on. The codes can be modified easily to run on actual gate-based (qubit) quantum processors. 
