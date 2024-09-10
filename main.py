import copy
import logging
import numpy as np,matplotlib.pyplot as plt
import pandas as pd
import healpy as hp
from astropy import units as u
from astropy import constants, coordinates, time
from typing import Iterable, Optional, Tuple, Union
from gaiaunlimited.utils import coord2healpix
import scanninglaw,scanninglaw.times
from astromet import sigma_ast
import sys,pickle
import astromet
import csv
import os
from stan_codes import ruwe_ecc,ruwe_circ,rv_ecc,rv_circ,im_ecc,im_circ,ruwe_ecc_dark,ruwe_circ_dark,rv_ecc_dark,rv_circ_dark,im_ecc_dark,im_circ_dark
import stan
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
scanninglaw.times.fetch(version='cog3_2020')
scanninglaw.times.fetch(version='dr3_nominal')
dr3_sl=scanninglaw.times.Times(version='dr3_nominal')
def find_obs(ra,dec):
	c=scanninglaw.source.Source(ra,dec,unit='deg')
	sl=dr3_sl(c, return_times=True, return_angles=True)
	ts=np.squeeze(np.hstack(sl['times'])).astype('double')
	angs = np.squeeze(np.hstack(sl['angles'])).astype('double')
	els = np.argsort(ts)
	ts = ts[els]/365.25+2010
	angs = angs[els]*np.pi/180
	p_factors = np.zeros(len(ts))
	for ii in range(len(ts)):
		[x,y,z] = np.array(gaia_position(ts[ii]-2016))/4.84814e-6
		f = x*np.sin(ra*np.pi/180)-y*np.cos(ra*np.pi/180)
		g = x*np.cos(ra*np.pi/180)*np.sin(dec*np.pi/180)+y*np.sin(ra*np.pi/180)*np.sin(dec*np.pi/180)-z*np.cos(dec*np.pi/180)
		p_factors[ii] = f*np.sin(angs[ii])+g*np.cos(angs[ii])
	return (ts,angs,p_factors)

def convert_julian_date(j_date):
	return j_date/365.25-4712
	
def system_pos(t_obs,ra,dec,ra_off,dec_off,pmra,pmdec,dist,m1,m2,l,period,eccentricity,inclination,Omega,omega,t_p):
	"""Calculate R.A. and Dec. offset of photocentre and barycentre"""
	ii = 0
	q = m2/m1
	semimajor_au = ((m1+m2)*period**2)**(1./3)
	semimajor = 1000*semimajor_au/dist
	ra_diffs_col = []
	dec_diffs_col = []
	ra_diffs_com = []
	dec_diffs_com = []
	[ra_diff,dec_diff] = radec_diff(t_obs,semimajor,period,q,l,eccentricity,inclination,Omega,omega,t_p)
	[x_gaia,y_gaia,z_gaia] = gaia_position(t_obs)
	ra_diff_com = ra_off+(x_gaia*np.sin(ra*np.pi/180)-y_gaia*np.cos(ra*np.pi/180))*1000./(dist*4.84814e-6)+t_obs*pmra
	dec_diff_com = dec_off+(x_gaia*np.cos(ra*np.pi/180)*np.sin(dec*np.pi/180)+y_gaia*np.sin(ra*np.pi/180)*np.sin(dec*np.pi/180)-z_gaia*np.cos(dec*np.pi/180))*1000./(dist*4.84814e-6)+t_obs*pmdec
	ra_diff_col = ra_diff_com+ra_diff
	dec_diff_col = dec_diff_com+dec_diff
	return (ra_diff_col,dec_diff_col,ra_diff_com,dec_diff_com)

def system_pos_multiple(t_obs,ra,dec,ra_off,dec_off,pmra,pmdec,dist,m1,m2,l,period,eccentricity,inclination,Omega,omega,t_p):
	all_ra_diffs = 0
	all_dec_diffs = 0
	for ii in range(len(m2)):
		q = m2[ii]/m1
		semimajor_au = ((m1+m2[ii])*period[ii]**2)**(1./3)
		semimajor = 1000*semimajor_au/dist
		ra_diffs_col = []
		dec_diffs_col = []
		ra_diffs_com = []
		dec_diffs_com = []
		[ra_diff,dec_diff] = radec_diff(t_obs,semimajor,period[ii],q,l[ii],eccentricity[ii],inclination,Omega,omega[ii],t_p[ii])
		all_ra_diffs+=ra_diff
		all_dec_diffs+=dec_diff
	[x_gaia,y_gaia,z_gaia] = gaia_position(t_obs)
	ra_diff_com = ra_off+(x_gaia*np.sin(ra*np.pi/180)-y_gaia*np.cos(ra*np.pi/180))*1000./(dist*4.84814e-6)+t_obs*pmra
	dec_diff_com = dec_off+(x_gaia*np.cos(ra*np.pi/180)*np.sin(dec*np.pi/180)+y_gaia*np.sin(ra*np.pi/180)*np.sin(dec*np.pi/180)-z_gaia*np.cos(dec*np.pi/180))*1000./(dist*4.84814e-6)+t_obs*pmdec
	ra_diff_col = ra_diff_com+all_ra_diffs
	dec_diff_col = dec_diff_com+all_dec_diffs
	return (ra_diff_col,dec_diff_col,ra_diff_com,dec_diff_com)
		
def radial_velocity(t_obs,m1,m2,l,period,eccentricity,inclination,Omega,omega,t_p):
	f = true_anomaly_calc(t_obs, period, eccentricity, t_p)
	q = m2/m1
	semimajor = ((m1+m2)*period**2)**(1./3)
	r = position(semimajor,f,eccentricity)
	[x,y,z] = position3d(r,f,inclination,Omega,omega)
	z0 = z*abs(q-l)/((1+l)*(1+q))
	t_diff = 0.001
	f = true_anomaly_calc(t_obs+t_diff, period, eccentricity, t_p)
	semimajor = ((m1+m2)*period**2)**(1./3)
	r = position(semimajor,f,eccentricity)
	[x,y,z] = position3d(r,f,inclination,Omega,omega)
	z1 = z*abs(q-l)/((1+l)*(1+q))
	rv = ((z1-z0)/t_diff)*1.496e8/(365.25*24*3600)
	return rv
	
def radial_velocity_multiple(t_obs,m1,m2,l,period,eccentricity,inclination,Omega,omega,t_p):
	all_rv = 0
	for ii in range(len(m2)):
		f = true_anomaly_calc(t_obs, period[ii], eccentricity[ii], t_p[ii])
		q = m2[ii]/m1
		semimajor = ((m1+m2[ii])*period[ii]**2)**(1./3)
		r = position(semimajor,f,eccentricity[ii])
		[x,y,z] = position3d(r,f,inclination,Omega,omega[ii])
		z0 = z*abs(q-l[ii])/((1+l[ii])*(1+q))
		t_diff = 0.001
		f = true_anomaly_calc(t_obs+t_diff, period[ii], eccentricity[ii], t_p[ii])
		semimajor = ((m1+m2[ii])*period[ii]**2)**(1./3)
		r = position(semimajor,f,eccentricity[ii])
		[x,y,z] = position3d(r,f,inclination,Omega,omega[ii])
		z1 = z*abs(q-l[ii])/((1+l[ii])*(1+q))
		rv = ((z1-z0)/t_diff)*1.496e8/(365.25*24*3600)
		all_rv+=rv
	return all_rv
	
def gaia_matrix(t_obs,ra,dec):
	[xG,yG,zG] = gaia_position(t_obs)
	nVar = 5
	A = np.zeros((2*len(t_obs),nVar))
	A[0:len(t_obs),0] = 1
	A[len(t_obs):2*len(t_obs),1] = 1
	A[0:len(t_obs),nVar-3] = (xG*np.sin(ra*np.pi/180)-yG*np.cos(ra*np.pi/180))/4.84814e-6
	A[len(t_obs):2*len(t_obs),nVar-3] = (xG*np.cos(ra*np.pi/180)*np.sin(dec*np.pi/180)+yG*np.sin(ra*np.pi/180)*np.sin(dec*np.pi/180)-zG*np.cos(dec*np.pi/180))/4.84814e-6
	A[0:len(t_obs),nVar-2] = t_obs
	A[len(t_obs):2*len(t_obs),nVar-1] = t_obs
	return A

def gaia_inverse(t_obs,ra,dec):
	A = gaia_matrix(t_obs,ra,dec)
	ATA = A.T @ A
	A_inv = np.linalg.inv(ATA) @ A.T
	return A_inv	
	
def gaia_matrix_AL(t_obs,scan_angle,ra,dec):
	[xG,yG,zG] = gaia_position(t_obs)
	nVar = 5
	A = np.zeros((len(t_obs),nVar))
	A[:,0] = np.sin(scan_angle)
	A[:,1] = np.cos(scan_angle)
	A[:,2] = (xG*np.sin(ra*np.pi/180)-yG*np.cos(ra*np.pi/180))*np.sin(scan_angle)/4.84814e-6+(xG*np.cos(ra*np.pi/180)*np.sin(dec*np.pi/180)+yG*np.sin(ra*np.pi/180)*np.sin(dec*np.pi/180)-zG*np.cos(dec*np.pi/180))*np.cos(scan_angle)/4.84814e-6
	A[:,3] = t_obs*np.sin(scan_angle)
	A[:,4] = t_obs*np.cos(scan_angle)
	return A

def gaia_inverse_AL(t_obs,scan_angle,ra,dec):
	A = gaia_matrix_AL(t_obs,scan_angle,ra,dec)
	ATA = A.T @ A
	A_inv = np.linalg.inv(ATA) @ A.T
	return A_inv
	
def gaia_quantities(t_obs,scan_angle,ra,dec,x,y):
	"""[xG,yG,zG] = gaia_position(t_obs)
	A = np.zeros((2*len(t_obs),3))
	A[0:len(t_obs),0] = (xG*np.sin(ra*np.pi/180)-yG*np.cos(ra*np.pi/180))/4.84814e-6
	A[len(t_obs):2*len(t_obs),0] = (xG*np.cos(ra*np.pi/180)*np.sin(dec*np.pi/180)+yG*np.sin(ra*np.pi/180)*np.sin(dec*np.pi/180)-zG*np.cos(dec*np.pi/180))/4.84814e-6
	A[0:len(t_obs),1] = t_obs
	A[len(t_obs):2*len(t_obs),2] = t_obs
	ATA = A.T @ A
	A_inv = np.linalg.inv(ATA) @ A.T"""
	A = gaia_matrix_AL(t_obs,scan_angle,ra,dec)
	A_inv = gaia_inverse_AL(t_obs,scan_angle,ra,dec)
	x_AL = x*np.sin(scan_angle)+y*np.cos(scan_angle)
	vec = A_inv @ x_AL
	parallax = vec[2]
	pmra = vec[3]
	pmdec = vec[4]
	return [vec[0],vec[1],parallax,pmra,pmdec]

def design_matrix(t_obs,scan_angle,parallax_factor):
	A = np.column_stack([np.sin(scan_angle),np.cos(scan_angle),parallax_factor,t_obs*np.sin(scan_angle),t_obs*np.cos(scan_angle)])
	return A

def init_offset(semimajor,period,q,l,eccentricity,inclination,Omega,omega,t_p):
	"""Photocentre offset from centre of mass at 2016.0"""
	mean_anomaly = -(t_p/period)*2*np.pi#initial_phase
	steps = 6
	eta = copy.copy(mean_anomaly)
	for ii in range(steps):
		eta-=(eta-eccentricity*np.sin(eta)-mean_anomaly)/(1-eccentricity*np.cos(eta))
	true_anomaly = 2*np.arctan(np.sqrt((1+eccentricity)/(1-eccentricity))*np.tan(eta/2))
	r = position(semimajor,true_anomaly,eccentricity)
	[x,y,z] = position3d(r,true_anomaly,inclination,Omega,omega)
	(x1,y1,x2,y2,xp,yp) = component_positions(y,x,q,l)
	return (xp,yp)
				
def component_positions(x,y,q,l):
	"""Positions of primary, secondary and photocentre relative to barycentre"""
	x1 = x*q/(1+q)
	y1 = y*q/(1+q)
	x2 = -x/(1+q)
	y2 = -y/(1+q)
	#xp = x/(1+q)
	#yp = y/(1+q)
	xp = x*abs(q-l)/((1+l)*(1+q))
	yp = y*abs(q-l)/((1+l)*(1+q))
	return (x1,y1,x2,y2,xp,yp)
	
def position3d(rad,true_anomaly,inclination,Omega,omega):
	"""Coordinates in 3D space, applying angles"""
	x = rad*np.cos(true_anomaly)
	y = rad*np.sin(true_anomaly)
	x_new = x*(np.cos(Omega*np.pi/180)*np.cos(omega*np.pi/180)-np.sin(Omega*np.pi/180)*np.cos(inclination*np.pi/180)*np.sin(omega*np.pi/180))+y*(-np.cos(Omega*np.pi/180)*np.sin(omega*np.pi/180)-np.sin(Omega*np.pi/180)*np.cos(inclination*np.pi/180)*np.cos(omega*np.pi/180))
	y_new = x*(np.sin(Omega*np.pi/180)*np.cos(omega*np.pi/180)+np.cos(Omega*np.pi/180)*np.cos(inclination*np.pi/180)*np.sin(omega*np.pi/180))+y*(-np.sin(Omega*np.pi/180)*np.sin(omega*np.pi/180)+np.cos(Omega*np.pi/180)*np.cos(inclination*np.pi/180)*np.cos(omega*np.pi/180))
	z_new = x*np.sin(inclination*np.pi/180)*np.sin(omega*np.pi/180)+y*np.sin(inclination*np.pi/180)*np.cos(omega*np.pi/180)
	return [x_new,y_new,z_new]	

def position(semimajor,true_anomaly,eccentricity):
	rad = semimajor*(1-eccentricity**2)/(1+eccentricity*np.cos(true_anomaly))
	return rad

def true_anomaly_calc(t, period, eccentricity, t_p):
	mean_anomaly = (((t-t_p)/period)*2*np.pi)%(2*np.pi)
	steps = 6
	eta = copy.copy(mean_anomaly)
	for ii in range(steps):
		eta-=(eta-eccentricity*np.sin(eta)-mean_anomaly)/(1-eccentricity*np.cos(eta))
	true_anomaly = 2*np.arctan(np.sqrt((1+eccentricity)/(1-eccentricity))*np.tan(eta/2))
	#return mean_anomaly
	return true_anomaly
		
def projected_position(t,semimajor,period,eccentricity,inclination,Omega,omega,t_p):
	"""Position on a 2D projection"""
	true_anomaly = true_anomaly_calc(t, period, eccentricity, t_p)
	r = position(semimajor,true_anomaly,eccentricity)
	[x,y,z] = position3d(r,true_anomaly,inclination,Omega,omega)
	#'Reference' direction is North for Campbell parameters so reverse x and y
	return (y,x)
	
def radec_diff(t,semimajor,period,q,l,eccentricity,inclination,Omega,omega,t_p):
	"""Difference between photocentre and barycentre on sky"""
	(x,y) = projected_position(t,semimajor,period,eccentricity,inclination,Omega,omega,t_p)
	(x1,y1,x2,y2,xp,yp) = component_positions(x,y,q,l)
	return (xp,yp)
	
def gaia_position(t):
	"""3D position of Gaia, with Sun at [0,0,0], in celestial coordinates"""
	t_p = 3./365.25
	Omega = 0
	i = 23.4
	e = 0.0167
	omega = 103.33284959908069
	p = 1
	a = 1.01*4.84814e-6
	phi_0 = t_p_to_init_phase(t_p,p)
	f = true_anomaly_calc(t,p,e,t_p)
	r = position(a,f,e)
	[x,y,z] = position3d(r,f,i,Omega,omega)
	return [x,y,z]

def t_p_to_init_phase(t_p,period):
	init_phase = -(t_p/period)*(2*np.pi)
	return init_phase
	
def system_pos_new(t,ra,dec,pmra,pmdec,dist,m1,m2,l,period,eccentricity,theta,phi,omega,t_p):
	q = m2/m1
	semimajor_au = ((m1+m2)*period**2)**(1./3)
	params = astromet.params()
	params.ra = ra
	params.dec = dec
	params.pmrac = pmra
	params.pmdec = pmdec
	params.parallax = 1000./dist
	x0,y0 = astromet.track(t,params)
	params.a = semimajor_au
	params.q = q
	params.l = l
	params.period = period
	params.e = eccentricity
	params.vtheta = theta*np.pi/180
	params.vphi = phi*np.pi/180
	params.vomega = omega*np.pi/180
	params.tperi = t_p+2016
	x,y = astromet.track(t,params)
	return (x,y,x0,y0)

def recover_gaia_quantities(t,x,y):
	year_els = [0]
	for ii in range(1,len(t)):
		if not np.floor(t[ii])==np.floor(t[ii-1]):
			year_els.append(ii)
	year_els.append(len(t))
	all_pmra = []
	all_pmdec = []
	all_plx = []
	for ii in range(len(year_els)-1):
		if year_els[ii+1]-year_els[ii]<10:
			continue
		t_year = t[year_els[ii]:year_els[ii+1]]
		x_year = x[year_els[ii]:year_els[ii+1]]
		y_year = y[year_els[ii]:year_els[ii+1]]
		all_pmra.append((x_year[-1]-x_year[0])/(t_year[-1]-t_year[0]))
		all_pmdec.append((y_year[-1]-y_year[0])/(t_year[-1]-t_year[0]))
		new_x = x_year-all_pmra[len(all_pmra)-1]*(t_year-2016)
		new_y = y_year-all_pmdec[len(all_pmdec)-1]*(t_year-2016)
		r_obs = np.sqrt(new_x**2+new_y**2)
		all_plx.append(np.max(r_obs))
	p_obs = np.mean(all_plx)
	pmra_obs = np.mean(all_pmra)
	pmdec_obs = np.mean(all_pmdec)
	p_err = np.std(all_plx)
	pmra_err = np.std(all_pmra)
	pmdec_err = np.std(all_pmdec)
	print(all_plx)
	return [p_obs,pmra_obs,pmdec_obs,p_err,pmra_err,pmdec_err]
	
def en_fit(R, err, w):
    y = 0
    nu = np.sum(w>=0.2)-5

    W = w/(err**2 + y)
    Q = np.sum(R**2 * W)

    W_prime = -w/(err**2 + y)**2
    Q_prime = np.sum(R**2 * W_prime)

    for i in range(4):
        W = w/(err**2 + y)
        Q = np.sum(R**2 * W)
        if (i==0)&(Q<=nu): break

        W_prime = -w/(err**2 + y)**2
        Q_prime = np.sum(R**2 * W_prime)

        y = y + (1-Q/nu)*Q/Q_prime

    return np.sqrt(y)

def gaia_params(x_obs,x_err,gaia_mat):
	weights = np.ones(len(x_obs))
	W = np.eye(len(x_obs))*weights/(x_err**2)
	C = np.linalg.inv(gaia_mat.T @ W @ gaia_mat)
	init_guess = C @ gaia_mat.T @ W @ x_obs
	R = x_obs - gaia_mat @ init_guess
	ISR = np.diff(np.percentile(R, [100.*1./6, 100.*5./6.]))[0]
	z = np.sqrt(R**2/((ISR/2)**2))
	w = np.where(z<2, 1, 1 - 1.773735*(z-2)**2 + 1.141615*(z-2)**3)
	weights = np.where(z<3, w, np.exp(-z/3))
	aen = 0
	for ii in range(10):
		W = np.eye(len(x_obs))*weights/((x_err**2+aen**2))
		C = np.linalg.inv(gaia_mat.T @ W @ gaia_mat)
		params = C @ gaia_mat.T @ W @ x_obs
		R = x_obs - gaia_mat @ params
		aen = en_fit(R, x_err, weights)
		z = np.sqrt(R**2/((x_err**2+aen**2)))
		w = np.where(z<2, 1, 1 - 1.773735*(z-2)**2 + 1.141615*(z-2)**3)
		weights = np.where(z<3, w, np.exp(-z/3))
		aen = en_fit(R, x_err, weights)
	C = np.linalg.inv(gaia_mat.T @ W @ gaia_mat)
	params = C @ gaia_mat.T @ W @ x_obs
	return params
	
def plx_angle(t,ra,dec,dist):
	r = 4.84e-6
	inc = 23.5
	omega = 103
	sin_dec = np.sin(dec*np.pi/180)-(r/dist)*np.sin(inc*np.pi/180)*np.sin(2*np.pi*t+omega*np.pi/180)
	dec_new = np.arcsin(sin_dec)
	dec_diff = dec_new*180/np.pi-dec
	tan_ra = (dist*np.cos(dec*np.pi/180)*np.sin(ra*np.pi/180)+r*np.cos(inc*np.pi/180)*np.sin(2*np.pi*t+omega*np.pi/180))/(dist*np.cos(dec*np.pi/180)*np.cos(ra*np.pi/180)+r*np.cos(2*np.pi*t+omega*np.pi/180))
	ra_new = np.arctan(tan_ra)
	ra_diff = ra_new*180/np.pi-ra
	ra_diff+=180*(ra_diff<-180)-180*(ra_diff>180)
	ang = np.arctan2(dec_diff,ra_diff)
	return (ra_diff*3.6e6,dec_diff*3.6e6,ang*180/np.pi)

def estimate_unit_weight_error(t_obs,scan_angle,parallax_factor,g_mag,ra_diffs,dec_diffs):
	A = np.column_stack([np.sin(scan_angle),np.cos(scan_angle),parallax_factor,t_obs*np.sin(scan_angle),t_obs*np.cos(scan_angle)])
	al_positions = ra_diffs*np.sin(scan_angle)+dec_diffs*np.cos(scan_angle)
	al_errors = sigma_ast(g_mag)
	C = np.linalg.solve(A.T @ A, np.eye(5))
	ACAT = A @ C @ A.T
	R = al_positions.T - (al_positions.T) @ ACAT
	T = len(t_obs)
	uwe = 0
	for ii in range(T):
		uwe += ((R[ii]/al_errors)**2) / (T - 5)
	uwe = np.sqrt(uwe)
	return uwe

def gaia_scan(t,t0,t1):
	[x_gaia,y_gaia,z_gaia] = gaia_position(t)
	gaia_ra = np.arctan2(y_gaia,x_gaia)
	gaia_dec = np.arcsin(z_gaia/np.sqrt(x_gaia**2+y_gaia**2+z_gaia**2))
	sun_theta = gaia_dec+np.pi/2
	sun_phi = gaia_ra+np.pi
	ax_ang = ((t+t0)*365.25/63.12)*2*np.pi
	prec_ang = -((t+t1)*365.25*24/6)*2*np.pi
	foll_ang = prec_ang+106.5*np.pi/180
	prec_x = np.cos(prec_ang)
	prec_y = np.sin(prec_ang)
	foll_x = np.cos(foll_ang)
	foll_y = np.sin(foll_ang)
	rot_x = np.sin(np.pi/4)*np.cos(ax_ang)*np.cos(sun_theta)*np.cos(sun_phi)-np.sin(np.pi/4)*np.sin(ax_ang)*np.sin(sun_phi)+np.cos(np.pi/4)*np.sin(sun_theta)*np.cos(sun_phi)
	rot_y = np.sin(np.pi/4)*np.cos(ax_ang)*np.cos(sun_theta)*np.sin(sun_phi)+np.sin(np.pi/4)*np.sin(ax_ang)*np.cos(sun_phi)+np.cos(np.pi/4)*np.sin(sun_theta)*np.sin(sun_phi)
	rot_z = -np.sin(np.pi/4)*np.cos(ax_ang)*np.sin(sun_theta)+np.cos(np.pi/4)*np.cos(sun_theta)
	rot_theta = np.arccos(rot_z)
	rot_phi = np.arctan2(rot_y,rot_x)
	rot_phi+=(rot_phi<0)*2*np.pi
	prec_x_new = prec_x*np.cos(rot_theta)*np.cos(rot_phi)-prec_y*np.sin(rot_phi)
	prec_y_new = prec_x*np.cos(rot_theta)*np.sin(rot_phi)+prec_y*np.cos(rot_phi)
	prec_z_new = -prec_x*np.sin(rot_theta)
	foll_x_new = foll_x*np.cos(rot_theta)*np.cos(rot_phi)-foll_y*np.sin(rot_phi)
	foll_y_new = foll_x*np.cos(rot_theta)*np.sin(rot_phi)+foll_y*np.cos(rot_phi)
	foll_z_new = -foll_x*np.sin(rot_theta)
	prec_ra = np.arctan2(prec_y_new,prec_x_new)*180/np.pi
	prec_ra+=360*(prec_ra<0)
	prec_dec = np.arcsin(prec_z_new)*180/np.pi
	foll_ra = np.arctan2(foll_y_new,foll_x_new)*180/np.pi
	foll_ra+=360*(foll_ra<0)
	foll_dec = np.arcsin(foll_z_new)*180/np.pi
	return [rot_theta*180/np.pi,rot_phi*180/np.pi,prec_ra,prec_dec,foll_ra,foll_dec]

def scanning_angles(t):
	[rot_theta,rot_phi,prec_ra,prec_dec,foll_ra,foll_dec] = gaia_scan(t,34.02/365.25,122.4/(60*24*365.25))
	[rot_theta0,rot_phi0,prec_ra0,prec_dec0,foll_ra0,foll_dec0] = gaia_scan(t-1./(3600*24*365.25),34.02/365.25,122.4/(60*24*365.25))
	print(prec_ra0,prec_ra,prec_dec0,prec_dec)
	prec_ang = np.arctan2(prec_dec-prec_dec0,prec_ra0-prec_ra)-np.pi/2
	foll_ang = np.arctan2(foll_dec-foll_dec0,foll_ra0-foll_ra)-np.pi/2
	return [prec_ang,foll_ang]

def thiele_innes(period,parallax,m1,m2,l,inclination,Omega,omega):
	"""Calculate Thiele-Innes parameters from Campbell parameters"""
	semimajor = ((m1+m2)*period**2)**(1./3)
	q = m2/m1
	a0 = semimajor*parallax*(q/(1+q)-l/(1+l))
	a1 = semimajor*parallax*(q/(1+q))
	A = a0*(np.cos(omega*np.pi/180)*np.cos(Omega*np.pi/180)-np.sin(omega*np.pi/180)*np.sin(Omega*np.pi/180)*np.cos(inclination*np.pi/180))
	B = a0*(np.cos(omega*np.pi/180)*np.sin(Omega*np.pi/180)+np.sin(omega*np.pi/180)*np.cos(Omega*np.pi/180)*np.cos(inclination*np.pi/180))
	F = -a0*(np.sin(omega*np.pi/180)*np.cos(Omega*np.pi/180)+np.cos(omega*np.pi/180)*np.sin(Omega*np.pi/180)*np.cos(inclination*np.pi/180))
	G = -a0*(np.sin(omega*np.pi/180)*np.sin(Omega*np.pi/180)-np.cos(omega*np.pi/180)*np.cos(Omega*np.pi/180)*np.cos(inclination*np.pi/180))
	C = (a1/parallax)*np.sin(omega*np.pi/180)*np.sin(inclination*np.pi/180)
	H = (a1/parallax)*np.cos(omega*np.pi/180)*np.sin(inclination*np.pi/180)
	return (A,B,F,G,C,H)

def gaia_diff_col(t_obs, ra, dec, pmra, pmdec, dist):
	[x_gaia,y_gaia,z_gaia] = gaia_position(t_obs)
	ra_diff_com = (x_gaia*np.sin(ra*np.pi/180)-y_gaia*np.cos(ra*np.pi/180))*1000./(dist*4.84814e-6)+t_obs*pmra
	dec_diff_com = (x_gaia*np.cos(ra*np.pi/180)*np.sin(dec*np.pi/180)+y_gaia*np.sin(ra*np.pi/180)*np.sin(dec*np.pi/180)-z_gaia*np.cos(dec*np.pi/180))*1000./(dist*4.84814e-6)+t_obs*pmdec
	return (ra_diff_com,dec_diff_com)

def best_chain(n_samples,n_chains,A,B,F,G,A_obs,B_obs,F_obs,G_obs):
	diffs = []
	for ii in range(n_chains):
		els = n_chains*np.arange(n_samples)+ii
		diffs.append(np.abs(np.median(A[els])-A_obs)+np.abs(np.median(B[els])-B_obs)+np.abs(np.median(F[els])-F_obs)+np.abs(np.median(G[els])-G_obs))
	diffs = np.array(diffs)
	return diffs.argmin()

def gaia_query(source_id):
	name = 'Gaia DR3 '+source_id
	#result_table = Simbad.query_object(name)
	result_table = Gaia.query_object(name,radius='10 arcsecond')
	all_ids = result_table['DESIGNATION'].tolist()
	el = all_ids.index(name)
	ra = result_table['ra'].tolist()[el]
	dec = result_table['dec'].tolist()[el]
	all_keys = ['ra','dec','parallax','pmra','pmdec']
	nKeys = len(all_keys)
	for ii in range(nKeys):
		all_keys.append(all_keys[ii]+'_error')
	all_keys.append('ruwe')
	all_keys.append('phot_g_mean_mag')
	all_keys.extend(['ra_dec_corr','ra_parallax_corr','ra_pmra_corr','ra_pmdec_corr','dec_parallax_corr','dec_pmra_corr','dec_pmdec_corr','parallax_pmra_corr','parallax_pmdec_corr','pmra_pmdec_corr'])
	params = {}
	for key in all_keys:
		params[key] = float(result_table[key][el])
	(t_obs,scan_angle,plx_factor) = find_obs(params['ra'],params['dec'])
	return (params,t_obs,scan_angle,plx_factor)

def get_fitted_params(samples):
    strings = []
    all_names = samples.constrained_param_names
    for ii in range(len(all_names)):
        if '.' in all_names[ii]:
            string = ''
            for jj in range(len(all_names[ii])):
                if all_names[ii][jj]=='.':
                    break
                string+=all_names[ii][jj]
            if string not in strings:
                strings.append(string)
        else:
            strings.append(all_names[ii])
    all_params = []
    if type(samples)==list:
        for ii in range(len(strings)):
            if strings[ii]=='cos_inclination':
                all_params.append(list(np.arccos(samples[0].get(strings[ii])[0])*180/np.pi))
            elif 'mega' in strings[ii]:
                all_params.append(list((samples[0].get(strings[ii])[0]*180/np.pi)%360))
            else:
                if len(samples[0].get(strings[ii]))==1:
                    all_params.append(list(samples[0].get(strings[ii])[0]))
                else:
                    all_params.append(list(samples[0].get(strings[ii])))
        for jj in range(1,len(samples)):
            for ii in range(len(strings)):
                if strings[ii]=='cos_inclination':
                    all_params[ii].extend(list(np.arccos(samples[jj].get(strings[ii])[0])*180/np.pi))
                elif 'mega' in strings[ii]:
                    all_params[ii].extend(list((samples[jj].get(strings[ii])[0]*180/np.pi)%360))
                else:
                    if len(samples[jj].get(strings[ii]))==1:
                        all_params[ii].extend(list(samples[jj].get(strings[ii])[0]))
                    else:
                        all_params[ii].extend(list(samples[jj].get(strings[ii])))
    else:
        for ii in range(len(strings)):
            if strings[ii]=='cos_inclination':
                all_params.append(np.arccos(samples.get(strings[ii])[0])*180/np.pi)
            elif 'mega' in strings[ii]:
                all_params.append((samples.get(strings[ii])[0]*180/np.pi)%360)
            else:
                if len(samples.get(strings[ii]))==1:
                    all_params.append(samples.get(strings[ii])[0])
                else:
                    all_params.append(samples.get(strings[ii]))
    return all_params

def create_sample_dict(samples,all_params):
    labels = []
    all_names = samples.constrained_param_names
    for ii in range(len(all_names)):
        change_names = ['cos_inclination','ra_real','dec_real','plx_real','pmra_real','pmdec_real']
        new_names = ['inc','ra_off','dec_off','parallax','pmra','pmdec']
        if all_names[ii] in change_names:
        	el = change_names.index(all_names[ii])
        	labels.append(new_names[el])
        else:
            if '.' in all_names[ii]:
                string = ''
                for jj in range(len(all_names[ii])):
                    if all_names[ii][jj]=='.':
                        break
                    string+=all_names[ii][jj]
                if string not in labels:
                    labels.append(string)
            else:
                labels.append(all_names[ii])
    samples_dict = {}
    for ii in range(len(all_params)):
        samples_dict[labels[ii]] = all_params[ii]
    samples_dict['initial_phase'] = samples_dict['initial_phase']%(2*np.pi)
    samples_dict['initial_phase'] = samples_dict['initial_phase']-2*np.pi*(samples_dict['initial_phase']>np.pi)
    samples_dict['t_peri'] = -samples_dict['initial_phase']*samples_dict['P']/(2*np.pi)
    return samples_dict
    
def find_companion(source_id,nSamps=4000,nChains=1,ast_error='auto',dark=False,circular=False,image_data=None,rv_data=None,init=None):
	(row,t_obs,scan_angle,plx_factor) = gaia_query(source_id)
	gaia_params = [row['ra'],row['dec'],row['parallax'],row['pmra'],row['pmdec']]
	gaia_errs = [row['ra_error'],row['dec_error'],row['parallax_error'],row['pmra_error'],row['pmdec_error']]
	gaia_corr = [row['ra_dec_corr'],row['ra_parallax_corr'],row['ra_pmra_corr'],row['ra_pmdec_corr'],row['dec_parallax_corr'],row['dec_pmra_corr'],row['dec_pmdec_corr'],row['parallax_pmra_corr'],row['parallax_pmdec_corr'],row['pmra_pmdec_corr']]
	ruwe_obs = row['ruwe']
	g_mag = row['phot_g_mean_mag']
	#print(gaia_params,gaia_errs,gaia_corr,ruwe_obs)
	gaia_params = np.array(gaia_params)
	gaia_errs = np.array(gaia_errs)
	gaia_corr = np.array(gaia_corr)
	[ra,dec,plx_obs,pmra_obs,pmdec_obs] = gaia_params
	[ra_err,dec_err,plx_err,pmra_err,pmdec_err] = gaia_errs
	cov_matrix = np.zeros((5,5))
	for ii in range(cov_matrix.shape[0]):
		cov_matrix[ii] = gaia_errs[ii]*gaia_errs[0:5]
		for jj in range(cov_matrix.shape[1]):
			if jj>ii:
				cov_matrix[ii,jj]*=gaia_corr[int(jj*(jj-1)/2)+ii]
			elif jj<ii:
				cov_matrix[ii,jj]*=gaia_corr[int(ii*(ii-1)/2)+jj]
	#(t,scan_angle,plx_factor) = find_obs(ra,dec)
	#pickle.dump((t,scan_angle,plx_factor),open('obs.pkl','wb'))
	#sys.exit()
	if dark:
		if circular:
			if image_data is None and rv_data is None:
				mod = ruwe_circ_dark
			elif rv_data is None:
				mod = im_circ_dark
			else:
				mod = rv_circ_dark
		else:
			if image_data is None and rv_data is None:
				mod = ruwe_ecc_dark
			elif rv_data is None:
				mod = im_ecc_dark
			else:
				mod = rv_ecc_dark
	else:
		if circular:
			if image_data is None and rv_data is None:
				mod = ruwe_circ
			elif rv_data is None:
				mod = im_circ
			else:
				mod = rv_circ
		else:
			if image_data is None and rv_data is None:
				mod = ruwe_ecc
			elif rv_data is None:
				mod = im_ecc
			else:
				mod = rv_ecc
	if ast_error=='auto':
		err = float(astromet.sigma_ast(g_mag))
		if np.isnan(err):
			err = 0.4
	else:
		err = ast_error
	model = mod.build_model(gaia_params,gaia_errs,gaia_corr,g_mag,ruwe_obs,0.1,t_obs,scan_angle,plx_factor,err,image_data,rv_data)
	if init is None:
		samples = model.sample(num_chains=nChains, num_samples=nSamps)
	else:
		samples = model.sample(num_chains=nChains, num_samples=nSamps,init=init)
	samples = create_sample_dict(samples,get_fitted_params(samples))
	return samples
	
source_id = '2305829918153638400' #Gaia source id or target name
samples = find_companion(source_id,ast_error='auto',dark=True)
pickle.dump(samples,open('samples_'+source_id+'.pkl','wb'))
