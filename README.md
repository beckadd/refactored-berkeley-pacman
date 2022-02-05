# refactored-berkeley-pacman
Is it hubris? maybe. Is the code awful? yes.

This is an attempt at refactoring the Berkeley Pacman project to meet modern standards of project code design. I chose to do this to not only make it easier to run (currently the project doesn't support anything beyond Python 3.7), but also to make it easier to read. The advent of Python typing and the application of better object oriented design will make this code easier to work with, without having to understand all the ins and outs of what makes the game actually run.

### Objectives
There are a few high-level goals I have with this project.

1. 1-click dev environment setup with VS Code.
2. Fully-typed classes.
3. Use of NumPy for array operations (of which there are several in this game).
4. Integrate the problem statements into the code.
5. Browser-based game rendering engine.
6. Intuitive TODO markers for locations student code is required.
7. Refactor code to be DRY! (Because it definitely isn't right now!)
8. Provide a pytest suite to better accommodate student testing (this autograder.py thing is a hot mess, and it hardly runs).
9. Setup a continuity plan - how will the project remain updated? How will coverage work? There needs to be some plan to maintain the project in the long term, so that sticking with an outdated version is not made desireable.
10. Fix those layout files - they're so bad y'all! How is anyone supposed to be able to interpret it besides a computer? I get that was somewhat of the intention - they're not designed to be human readable - but we can implement better layouts in the browser application to illustrate them better.

### Why a browser-based engine, you ask?
This all comes down to a separation of concerns. The fact is, students will never interface with the rendering code, and will only focus on writing the code that will affect the movement of pacman as well as grabbing from the game state. While this breaks up the interface a bit, it makes the code a lot more managable. The code will provide a list of events as a json file, and will then run the game based off of the events passed to the json file. This also decouples the game's renderer from the game's decisions, making it possible to completely rewrite either the renderer or the engine as a drop-in replacement should the need arise.
