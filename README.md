# room-modal-optimizer
Tool for optimizing small-room geometry by evaluating modal response using genetic algorithms acoustic simulation tools like finite element methods.

The code is organized based on a genetic algorithm that runs the following pipeline as its fitness function:

1. Define the geometry of a given room and discretize into finite elements
2. Simulate the room with finite elements methods
3. Evaluate the rooms performance

Each of the steps in the pipeline have their corresponding class.
