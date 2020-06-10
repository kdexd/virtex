virtex.factories
================

.. raw:: html

    <hr>

.. First only include the top-level module, and base class docstrings.

.. automodule:: virtex.factories
    :no-members:

.. autoclass:: virtex.factories.Factory


------------------------------------------------------------------------------

Dataloading-related Factories
-----------------------------

.. autoclass:: virtex.factories.TokenizerFactory
    :members: from_config

.. autoclass:: virtex.factories.ImageTransformsFactory
    :members: from_config

.. autoclass:: virtex.factories.PretrainingDatasetFactory
    :members: from_config

.. autoclass:: virtex.factories.DownstreamDatasetFactory
    :members: from_config

------------------------------------------------------------------------------

Modeling-related Factories
--------------------------

.. autoclass:: virtex.factories.VisualBackboneFactory
    :members: from_config

.. autoclass:: virtex.factories.TextualHeadFactory
    :members: from_config

.. autoclass:: virtex.factories.PretrainingModelFactory
    :members: from_config

------------------------------------------------------------------------------

Optimization-related Factories
------------------------------

.. autoclass:: virtex.factories.OptimizerFactory
    :members: from_config

.. autoclass:: virtex.factories.LRSchedulerFactory
    :members: from_config
