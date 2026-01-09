"""Contains the name for the logger of DerivKit modules.

``derivkit`` uses a simple logging system based on the
`Logging <https://docs.python.org/3/library/logging.html>`__ standard library.
Logging messages are grouped in different levels:

* ``INFO``: An indication that things are working as expected.
* ``WARNING``: An indication that something unexpected
    happened which may require attention.

By default, only messages of level ``WARNING`` are displayed.

Calling applications can configure the format and log level of the displayed messages
by `Configuring Logging <https://docs.python.org/3/howto/logging.html#configuring-logging>`__
for ``derivkit.logger.derivkit_logger``, e.g.::

    >>> import logging
    >>> logging.basicConfig(
    ...     level=logging.INFO,
    ...     format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    ... )
"""
import logging

logger_name = "derivkit"
derivkit_logger = logging.getLogger(logger_name)
