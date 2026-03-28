Configuration
=============

``Athelas`` input decks are written in `Lua <https://www.lua.org/>`_. Each input
file defines a table ``config`` and returns it at the end of the file — this
exported table is what ``Athelas`` reads at startup to configure the problem,
physics, and numerical methods. Basically, we are constructing a dictionary.

.. code-block:: lua

    local config = {}

    config.problem = { ... }
    config.physics = { ... }
    -- and so on

    return config  -- required

The full set of recognised keys, their types, defaults, and descriptions are
documented in :doc:`schema_reference`, which is auto-generated from the schema
file.

Using Lua in input decks
------------------------

By using Lua we have access to a configuration *language* instead of a format.
Because input decks are valid Lua source files, users have access to the full
convenience of a scripting language when defining their problem. Local variables
are a particularly useful tool for clarity and correctness — physical constants,
unit conversions, and derived quantities can be defined once and referenced
throughout the file.

.. code-block:: lua

    local DAY = 86400.0  -- seconds

    local config = {}

    config.problem = {
        name      = "supernova",
        ...
        t_end     = 150.0 * DAY,
        ...
    }

    return config

Standard Lua comments apply: ``--`` for a single line, ``--[[ ... ]]`` for a
block. The ``math`` library is available, so expressions like
``math.pi`` or ``math.exp(x)`` are valid in the config.

We can use existing Lua input decks to define new ones that use the same 
problem generator. For example, consider the Leblanc problem:

.. code-block:: lua

    local config = dofile("../inputs/sod.lua")

    -- Override key pieces
    config.problem.params = {
      vL = 0.0,
      vR = 0.0,
      rhoL = 1.0,
      rhoR = 1.0e-3,
      pL = 0.066666667,
      pR = 0.666666667e-10,
      x_d = 3.0,
    }

    config.basis.nnodes = 2
    config.time.integrator = "EX_SSPRK54"
    config.eos.gamma = 5.0 / 3.0

    return config

This loads the complete ``sod.lua`` input deck and overrides necessary keys to
form the Leblanc shock tube problem.

.. note::
   Handling configuration with Lua enables a lot of potential capabilities. 
   While not yet implemented, this enables custom output, initial conditions, 
   enabling/disabling of physics packages, and more.

.. _schema-tools:

Schema tools
------------

A template or schema lives in ``inputs/schema.lua``. This defines the possible
values that can be taken and defines whether the key is required, optional, 
or dependent on another key (e.g., opacity is required to be specified when 
radiation is enabled). Here we define *docstrings* for documentation.
Tools are provided for working with the schema.


Interactive explorer
~~~~~~~~~~~~~~~~~~~~

``scripts/python/athelas_tools/src/athelas_src/config_explorer.py`` 
is a terminal UI for browsing the schema interactively.
This can be used to explore the available configuration options.


Formatting
~~~~~~~~~~~~~~~~~~~~

The Lua input deck format is managed with `stylua`. The settings are in 
``${ATHELAS_ROOT}/.stylua.toml``. Compliance with this format is checked on
every PR. With ``stylua`` installed you may simply:

.. code-block:: sh

   stylua input.lua

or you may configure your text editor to format on save.
