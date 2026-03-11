-- NIST/CODATA/Astropy Compatible Constants in CGS Units
-- Length: cm | Mass: g | Time: s | Energy: erg

local constants = {
  -- Fundamental Physical Constants
  c = 2.99792458e10, -- Speed of light [cm/s]
  G = 6.67430e-8, -- Gravitational constant [cm^3 g^-1 s^-2]
  h = 6.62607015e-27, -- Planck constant [erg s]
  hb = 1.054571817e-27, -- Reduced Planck (h-bar) [erg s]
  kb = 1.380649e-16, -- Boltzmann constant [erg/K]
  sigma_sb = 5.670374419e-5, -- Stefan-Boltzmann [erg cm^-2 s^-1 K^-4]

  -- Atomic / Particle
  m_p = 1.672621923e-24, -- Proton mass [g]
  m_e = 9.1093837015e-28, -- Electron mass [g]
  ev = 1.602176634e-12, -- Electron volt [erg]

  -- Solar / Astronomical (IAU 2015)
  M_sun = 1.98840987e33, -- Solar mass [g]
  R_sun = 6.957e10, -- Solar radius [cm]
  L_sun = 3.828e33, -- Solar luminosity [erg/s]

  -- Distance
  au = 1.495978707e13, -- Astronomical Unit [cm]
  pc = 3.085677581e18, -- Parsec [cm]
  ly = 9.460730472e17, -- Light year [cm]
}

return constants
