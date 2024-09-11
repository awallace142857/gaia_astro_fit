Program using Bayesian inference with stan to constrain orbital parameters of potential companions using Gaia astrometry and other methods.  Requires pystan version 3.9.0.

main.py contains code to run analysis on single source. <br />
stan_codes contains models for specific cases depending on input. <br />

Main function: find_companion <br />
Takes source_id as input (string of numbers) and from this, finds the observed position, parallax, proper motion, ruwe etc. <br />
Optional inputs: <br />
nSamps: number of samples (default 4000) <br />
ast_error: astrometric uncertainty (detault 'auto', ast_error taken from astromet.sigma_ast) <br />
dark: whether the companion's brightness is negligible (default False) <br />
circular: are we assuming a circular orbit? (default False) <br />
image_data: np array with position of companion with respect to primary (default None) <br />
            1st column: time in years <br />
            2nd column: ra offset in mas <br />
            3rd column: dec offset in mas <br />
rv_data: np array with radial velocity data (default None) <br />
            1st column: time in years <br />
            2nd column: radial velocity in km/s <br />
init: initial guess of parameters as a dictionary <br />
<br />
Returns: <br />
Posterior samples of: <br />
masses <br />
period <br />
eccentricity (if not circular) <br />
inclination <br />
longitude of ascending node <br />
argument of periapsis <br />
time of periastron <br />
true position offset <br />
true parallax <br />
true proper motion <br />
