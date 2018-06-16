## Ultimate Tic-Tac-Toe AI Project

# Game Description

This is an extension of the popular Tic-Tac-Toe. 
Unlike the regular one, which is fairly trivial, this one has a certain twist that gives room for strategic planning.

The 9x9 grid can be throught of as a global 3x3 grid, each being a local 3x3 grid.

For the first move, any of the 81 cells can be picked.

But the next valid local grid is decided by which cell in the previously played local grid was marked.

To win, you must win the global grid recursively.

# Ambiguities

Many sources describe the above gameplay. But rules for some cases aren't very standardized.
The following are followed in this implementation:

- In case the selected local grid happens to be already won, or in draw, the player can choose any of the local grids with playable cells, including the original grid.
- Since this makes it possible for a grid previously won by a player to be won again by the other, only the first status will be considered.


