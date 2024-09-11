Program using Bayesian inference with stan to constrain orbital parameters of potential companions using Gaia astrometry and other methods.  Requires pystan version 3.9.0.

main.py contains code to run analysis on single source
stan_codes contains models for specific cases depending on input

Main function: find_companion
Takes source_id as input (string of numbers) and from this, finds the observed position, parallax, proper motion, ruwe etc.
Optional inputs: 
nSamps: number of samples (default 4000)
ast_error: astrometric uncertainty (detault 'auto', ast_error taken from astromet.sigma_ast)
dark: whether the companion's brightness is negligible (default False)
circular: are we assuming a circular orbit? (default False)
image_data: np array with position of companion with respect to primary (default None)
            1st column: time in years
            2nd column: ra offset in mas
            3rd column: dec offset in mas
rv_data: np array with radial velocity data (default None)
            1st column: time in years
            2nd column: radial velocity in km/s
init: initial guess of parameters as a dictionary

Returns:
Posterior samples of:
masses
period
eccentricity (if not circular)
inclination
longitude of ascending node
argument of periapsis
time of periastron
true parallax
true proper motion
