.. _contributing:

############
Contributing
############

Contributions in any shape or form are appreciated.
Below are some minimal guidelines to get started.

Development of ``derivkit`` is organised around `the Github repository <https://github.com/derivkit/derivkit>`__.
Contributing usually requires `setting up an account <https://github.com/signup>`__ on Github.
No worries, it's free of charge!

When submitting contributions, please write in clear and correct English using full sentences.
Be concise and avoid unnecessarily formulaic descriptions.

***********************
Submitting bugs or code
***********************

Submitting a bug report or feature request can be done by `opening an issue on Github <https://github.com/derivkit/derivkit/issues/new/choose>`__.
In the case of a bug report please make sure to

- describe the expected behaviour,
- describe the actual behaviour,
- the version(s) of ``derivkit`` that produce the bug,
- any specific environmnent details.

In the case of a feature please make sure to

- describe the feature,
- describe the need of the feature.

Submitting a code contribution can be done by `opening a pull request on Github <https://github.com/derivkit/derivkit/compare>`__.
In the pull request, please make sure of the following:

- The description of the pull request contains a high-level overview of what is implemented in the contribution.
- The description describes the reason for the addition.
  Specifically, it describes what problem it is supposed to solve.
  If the pull requests resolves a bug or implements a feature for which an issue exists, make sure to refer to this.
- The ``derivkit`` workflows completed successfully.
  The workflows will activate when a pull request is created, but can also be run locally on your computer as described below

******************************
Running ``derivkit`` workflows
******************************

``derivkit`` uses `tox <https://tox.wiki>`__ to run its workflows.
It is installed along with  ``derivkit`` itself.

All workflows can be run consecutively by simply calling tox from the project root directory::

  tox

Specific workflows can also be run in isolation.
The following workflows are provided:

=======
Linting
=======

Code for ``derivkit`` must comply with `PEP-8 <https://peps.python.org/pep-0008/>`__.
Comments and docstrings must be compatible with `Google-style comments and docstrings <https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings>`__.

The linting workflow can be run using::

  tox -e lint

=============
Documentation
=============

Documentation written in `reStructuredText <https://docutils.sourceforge.io/rst.html>`__ (rST).
Code documentation is generated from docstrings, and can be generated automatically::

  tox -e docs

Newly created rST files may need to be added to the appropriate table of contents files by hand.

=======
Testing
=======

Contributions that contain new code must include tests in the appropriate files in the ``tests`` directory.
The test suite can be run locally by simply running tox on the command line::

  tox -m test

Note that this will attempt to run the test suite for all supported Python version.
It will automatically skip versions which aren't locally available.
If the test suite must be run for a specific version of Python that specific environment must be called.
For example, to test against Python 3.13::

  tox -e py313
