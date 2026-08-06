-- Reusable xmake rules, kept free of any one project's assumptions so the
-- directory can be copied into another project as-is. See README.md.
--
--   includes("xmake/toolkit")
--
-- defines the rules toolkit.modes, toolkit.sanitizers, and toolkit.provenance,

includes("modes.lua")
includes("sanitizers.lua")
includes("provenance.lua")
