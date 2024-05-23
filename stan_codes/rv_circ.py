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

def get_fitted_params(samples):
    strings = ['m1','m2','P','e','cos_inclination','Omega','omega','t_peri','ra_real','dec_real','plx_real','pmra_real','pmdec_real','ruwe']
    all_params = {}
    for ii in range(len(strings)):
        if strings[ii]=='cos_inclination':
            all_params[strings[ii]] = np.arccos(samples.get(strings[ii])[0])*180/np.pi
        elif 'mega' in strings[ii]:
            all_params[strings[ii]] = (samples.get(strings[ii])[0]*180/np.pi)%360
        else:
            all_params[strings[ii]] = samples.get(strings[ii])[0]
    return all_params
    
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

def mass_from_mag(abs_mag):
    m_sun = 4.83
    m = 10**(0.25*(m_sun-abs_mag))
    if m<0.43:
        m = ((1./0.23)**(1./2.3))*10**((m_sun-abs_mag)/2.3)
    elif m>2:
        m = ((1./1.4)**(1./3.5))*10**((m_sun-abs_mag)/3.5)
    return m
program_code = """functions {
    real photocenter_scalar(real q, real l) {
        return abs(q - l)/((1 + l) * (1 + q));
    }    
    vector true_anomaly_calc(int T, vector t, real P, real t_peri) {
        /* 
        Calculate the true anomaly.
        
        :param T:
            The number of observations.
        :param t:
            The observed times [year].
        :param P:
            Orbital period [year].                
        :param t_peri:
            Periastron time [year].
        */
        vector[T] phase = 2 * pi() * ((t - t_peri) / P);
        return phase;
    }
    vector radial_velocity(int T, vector t, real a, real q, real l, real P, real cos_inclination, real Omega_, real omega_,real t_peri) {
    	vector[T] f = true_anomaly_calc(T, t, P, t_peri);
    	vector[T] x = a .* cos(f);
        vector[T] y = a .* sin(f);
        vector[T] z_new = x*sqrt(1-pow(cos_inclination,2))*sin(omega_) + y*sqrt(1-pow(cos_inclination,2))*cos(omega_);
        vector[T] z0 = z_new.*photocenter_scalar(q, l);
        real t_diff = 0.001;
        f = true_anomaly_calc(T, t+t_diff, P, t_peri);
    	x = a .* cos(f);
        y = a .* sin(f);
        z_new = x*sqrt(1-pow(cos_inclination,2))*sin(omega_) + y*sqrt(1-pow(cos_inclination,2))*cos(omega_);
        vector[T] z1 = z_new.*photocenter_scalar(q, l);
        real rv_scalar = 1.496*pow(10,8)/(365.25*24*3600);
        vector[T] rv = rv_scalar*(z1-z0)./t_diff;
        return rv;
    }
    vector[] position_2d(int T, vector true_anomaly, vector rad, real cos_inclination, real Omega_, real omega_) {
        /*
        Calculate the 2D position of a component.

        :param T:
            The number of observations.
        :param true_anomaly:
            The true anomaly [radians].
        :param rad:
            The radial position [arcsec].
        :param cos_inclination:
            The cosine of the inclination angle.
        :param Omega_:
            The longitude of the ascending node [radians].
        :param omega_:
            The argument of periastron [radians].
        */
        vector[T] x = rad .* cos(true_anomaly);
        vector[T] y = rad .* sin(true_anomaly);
        vector[T] x_new = x*(cos(Omega_)*cos(omega_)-sin(Omega_)*cos_inclination*sin(omega_)) + y*(-cos(Omega_)*sin(omega_)-sin(Omega_)*cos_inclination*cos(omega_));
        vector[T] y_new = x*(sin(Omega_)*cos(omega_)+cos(Omega_)*cos_inclination*sin(omega_)) + y*(-sin(Omega_)*sin(omega_)+cos(Omega_)*cos_inclination*cos(omega_));
        //vector[T] z_new = x*sin(inclination)*sin(omega_) + y*sin(inclination)*cos(omega_);
        return {x_new, y_new};
    }
    vector[] projected_position_2d(int T, vector t, real a, real P, real cos_inclination, real Omega_, real omega_, real t_peri) {
        /*
        Calculate the projected position in the plane of the sky.

        :param t:
            The observed times [year].
        :param a:
            The semi-major axis [arcsec].
        :param P:
            Orbital period [year].
        :param cos_inclination:
            The cosine of the inclination angle.
        :param Omega_:
            The longitude of the ascending node [radians].
        :param omega_:
            The argument of periastron [radians].
        :param t_peri:
            Periastron time [year].
        */
        vector[T] true_anomaly = true_anomaly_calc(T, t, P, t_peri);
        vector[T] xyz[2];
        vector[T] rad = a.*true_anomaly./true_anomaly;
        xyz = position_2d(T, true_anomaly, rad, cos_inclination, Omega_, omega_);
        return {xyz[2], xyz[1]};
    }
    real[] initial_position(real a, real P, real q, real l, real cos_inclination, real Omega_, real omega_, real t_peri) {
        vector[1] xy[2] = projected_position_2d(1, rep_vector(0, 1), a, P, cos_inclination, Omega_, omega_, t_peri);
        real s = photocenter_scalar(q, l);
        return {xy[1][1] * s, xy[2][1] * s};
    }
    vector[] source_position(int T, vector t, real a, real P, real q, real l, real cos_inclination, real Omega_, real omega_, real t_peri) {
        vector[T] xy[2] = projected_position_2d(T, t, a, P, cos_inclination, Omega_, omega_, t_peri);
        real s = photocenter_scalar(q, l);
        return {xy[1] * s, xy[2] * s};
    }    
    vector[] system_pos(int T, vector t, vector gaia_ra_col, vector gaia_dec_col, real q, real l,real a, real P, real cos_inclination, real Omega_, real omega_, real t_peri) {
        real p0[2] = initial_position(a, P, q, l, cos_inclination, Omega_, omega_, t_peri);
        vector[T] p1[2] = source_position(T, t, a, P, q, l, cos_inclination, Omega_, omega_, t_peri);
        vector[T] ra_diff_col = gaia_ra_col + (p1[1]);
        vector[T] dec_diff_col = gaia_dec_col + (p1[2]);
        return {ra_diff_col, dec_diff_col};
    }
    vector downweight(int T, vector R, real al_err, real aen) {
    	vector[T] z = sqrt((R.*R)./(pow(al_err,2) + pow(aen,2)));
    	vector[T] w;
    	for (n in 1:T) {
    		if (z[n]<2)
    			w[n] = 1;
    		else
    			w[n] = 1 - 1.773735*pow(z[n]-2,2) + 1.141615*pow(z[n]-2,3);
    		if (z[n]>=3)
    			w[n] = exp(-z[n]/3);
    	}
    	return w;
    }
    real flux_from_mass(real mass) {
    	real T_sun = 5800.0;
    	real temp = T_sun*mass;
    	real freq = 3*pow(10,8)/(5*pow(10,-7));
    	real h = 6.63*pow(10,-34);
    	real k_B = 1.38*pow(10,-23);
    	if (temp==0)
        	return 0;
    	else
        	return 1./((h*freq)/(k_B*temp)-1);
    }
    real mag_from_mass(real m1, real m2, real parallax) {
    	real flux_ratio = (flux_from_mass(m1)+flux_from_mass(m2))/flux_from_mass(1.0);
    	real mag_sun = 4.83;
    	real mag = mag_sun-2.5*log10(flux_ratio)+5*log10(100./parallax);
    	return mag;
    }
    real excess_noise(int T, vector R, real al_err, vector w) {
    	real y=0;
    	real nu=0;
    	for (n in 1:T) {
    		if (w[n]>=0.2)
    			nu+=1;
    	}
    	nu-=5;
    	vector[T] W = w/(pow(al_err,2)+y);
    	vector[T] W_prime = -w/pow((pow(al_err,2)+y),2);
		real Q = sum(pow(R,2).*W);
		real Q_prime = sum(pow(R,2).*W_prime);
		for (n in 1:4) {
			W = w/(pow(al_err,2)+y);
			W_prime = -w/pow((pow(al_err,2)+y),2);
			Q = sum(pow(R,2).*W);
			Q_prime = sum(pow(R,2).*W_prime);
			if (n==1)
				if (Q<=nu)
					break;
			y += (1-Q/nu)*Q/Q_prime;
		}
		return sqrt(y);
    }
    vector gaia_params(int T, vector x_AL, real al_err, matrix gaia_mat) {
    	vector[T] weights = rep_vector(1,T);
    	matrix[T,T] W = diag_matrix(weights)/(pow(al_err,2));
    	matrix[5,5] C = inverse(gaia_mat'*W*gaia_mat);
    	vector[5] params = C*gaia_mat'*W*x_AL;
    	vector[T] R = x_AL-gaia_mat*params;
    	vector[T] R_sort = sort_asc(R);
    	real ISR = R_sort[5*T/6]-R_sort[T/6];
    	weights = downweight(T, R, ISR./2, 0);
    	real aen = 0.0;
    	for (n in 1:10) {
    		W = diag_matrix(weights)/(pow(al_err,2)+pow(al_err,2));
    		C = inverse(gaia_mat'*W*gaia_mat);
    		params = C*gaia_mat'*W*x_AL;
    		R = x_AL-gaia_mat*params;
    		aen = excess_noise(T, R, al_err, weights);
    		weights = downweight(T, R, al_err, aen);
    		aen = excess_noise(T, R, al_err, weights);
    	}
    	C = inverse(gaia_mat'*W*gaia_mat);
    	params = C*gaia_mat'*W*x_AL;
    	return params;
    }
    real reduced_unit_weight_error(int T, vector t, vector gaia_ra_col, vector gaia_dec_col, real q, real l, real a, real P, real cos_inclination, real Omega_, real omega_, real t_peri, vector sin_scan_angle, vector cos_scan_angle, real al_err, matrix ACAT) {
        //real t_peri = -initial_phase * P / (2*pi());
        vector[T] pos[2] = system_pos(T, t, gaia_ra_col, gaia_dec_col, q, l, a, P, cos_inclination, Omega_, omega_, t_peri);
        vector[T] al_pos = pos[1] .* sin_scan_angle + pos[2] .* cos_scan_angle;
        vector[T] R = (al_pos' - al_pos' * ACAT)';
        real ruwe = sqrt(sum(pow(R/al_err, 2)) / (T - 5));
        if (is_nan(ruwe)) {
            print("q=", q, ";a=", a,";P=",P,";cosi=",cos_inclination,";Omega=",Omega_,";omega=",omega_);//,";phi0=",initial_phase);
        }
        return ruwe;
        return sqrt(sum(pow(R/al_err, 2)) / (T - 5));
    }
    vector[] gaia_pos(int T, real ra_off, real dec_off, real parallax, real pmra, real pmdec, matrix gaia_mat) {
    	vector[5] obs_vec = [ra_off,dec_off,parallax,pmra,pmdec]';
    	vector[2*T] diffs = gaia_mat*obs_vec;
    	vector[T] ra_diff_com = diffs[1:T];
    	vector[T] dec_diff_com = diffs[T+1:2*T];
    	return {ra_diff_com,dec_diff_com};
    }
    vector gaia_obs(int T, vector sin_scan_angle, vector cos_scan_angle, vector ra_diff, vector dec_diff, matrix inv_gaia){
    	vector[T] pos_AL;
    	pos_AL = ra_diff.*sin_scan_angle + dec_diff.*cos_scan_angle;
    	vector[5] obs_vals = inv_gaia*pos_AL;
    	return obs_vals;
    }
}
data {
    int T;
    vector[T] t;
    vector[T] scan_angle;
    vector[T] ast_noise;
    matrix[T, T] ACAT;
    real al_err;
    real parallax;
    real ra;
    real dec;
    real pmra;
    real pmdec;
    matrix[2*T,5] gaia_mat;
    matrix[T,5] gaia_mat_AL;
    matrix[5,T] inv_gaia;
    matrix[5,5] covar_matrix;
    real ruwe_obs;
    real ruwe_err;
    real m1_mean;
    real g_mag;
    real mag_err;
    int T2;
    vector[T2] rv_vals;
    vector[T2] rv_t;
    vector[T2] rv_errs;
}
transformed data {  
    vector[T] sin_scan_angle = sin(scan_angle);
    vector[T] cos_scan_angle = cos(scan_angle);
}
parameters {
    real<lower=0.08, upper=3> m1; // solar masses
    real<lower=0.002, upper=m1> m2; // solar masses
    real<lower=0.1, upper=40> P;
    real initial_phase;
    real<lower=-1, upper=+1> cos_inclination;
    real Omega;
    real omega;
    real plx_real;
    real pmra_real;
    real pmdec_real;
    real<lower=-20, upper=20> ra_real;
    real<lower=-20, upper=20> dec_real;
    
}
transformed parameters {
	real q = m2/m1;
	real dist = 1000./plx_real;
	real a = 1000 * pow((m1+m2)*pow(P, 2), 1./3) / dist;
	real l = pow(q,4);
	real t_peri = -initial_phase * P / (2*pi());
}
model {
    m1 ~ normal(m1_mean, 0.8);
    {
        real a0 = a*(q/(1+q)-l/(1+l)); // l = 0 for a dark companion
        real a1 = a*(q/(1+q));
        vector[T] gaia_diffs[2] = gaia_pos(T, ra_real, dec_real, plx_real, pmra_real, pmdec_real, gaia_mat);
        vector[T] total_diffs[2] = system_pos(T, t, gaia_diffs[1], gaia_diffs[2], q, l, a, P, cos_inclination, Omega, omega, t_peri);
        vector[T] x_AL = total_diffs[1].*sin_scan_angle +  total_diffs[2].*cos_scan_angle;
        x_AL = x_AL + ast_noise;
        vector[5] obs_vals = gaia_params(T, x_AL, al_err, gaia_mat_AL);
        vector[5] y = [0,0,parallax,pmra,pmdec]';
        y ~ multi_normal(obs_vals,covar_matrix);
        vector[T2] rv_sim = radial_velocity(T2, rv_t, a*dist/1000, q, l, P, cos_inclination, Omega, omega, t_peri);
        rv_vals ~ multi_normal(rv_sim,diag_matrix(rv_errs.*rv_errs));
        real ruwe = sqrt(sum(pow(gaia_mat_AL*obs_vals-x_AL,2)/(pow(al_err,2)*(T-5))));//reduced_unit_weight_error(T, t, gaia_diffs[1], gaia_diffs[2], q, l, a, P, e, cos_inclination, Omega, omega, t_peri, sin_scan_angle, cos_scan_angle, al_err, ACAT);
        ruwe_obs ~ normal(ruwe, ruwe_err);
        real mag_sim = mag_from_mass(m1,bright*m2,plx_real);
        g_mag ~ normal(mag_sim, mag_err);
    }
}

generated quantities {
	vector[5] stan_params;
	{
		vector[T] gaia_diffs[2] = gaia_pos(T, ra_real, dec_real, plx_real, pmra_real, pmdec_real, gaia_mat);
		vector[T] total_diffs[2] = system_pos(T, t, gaia_diffs[1], gaia_diffs[2], q, l, a, P, cos_inclination, Omega, omega, t_peri);
        vector[T] x_AL = total_diffs[1].*sin_scan_angle +  total_diffs[2].*cos_scan_angle;
		stan_params = gaia_params(T, x_AL, al_err, gaia_mat_AL);
	}
    real ruwe;
    {
        vector[T] gaia_diffs[2] = gaia_pos(T, ra_real, dec_real, plx_real, pmra_real, pmdec_real, gaia_mat);
		vector[T] total_diffs[2] = system_pos(T, t, gaia_diffs[1], gaia_diffs[2], q, l, a, P, cos_inclination, Omega, omega, t_peri);
        vector[T] x_AL = total_diffs[1].*sin_scan_angle +  total_diffs[2].*cos_scan_angle;
        x_AL = x_AL + ast_noise;
        vector[5] obs_vals = gaia_params(T, x_AL, al_err, gaia_mat_AL);
        ruwe = sqrt(sum(pow(gaia_mat_AL*obs_vals-x_AL,2)/(pow(al_err,2)*(T-5))));//reduced_unit_weight_error(T, t, diffs[1], diffs[2], q, l, a, P, e, cos_inclination, Omega, omega, t_peri, sin_scan_angle, cos_scan_angle, al_err, ACAT);
    }
    array[4] real ti;
    {
        real a0 = a*(q/(1+q)-l/(1+l));
        ti[1] = a0*(cos(omega)*cos(Omega) - sin(omega)*sin(Omega)*cos_inclination);
        ti[2] = a0*(cos(omega)*sin(Omega) + sin(omega)*cos(Omega)*cos_inclination);
        ti[3] = -a0*(sin(omega)*cos(Omega) + cos(omega)*sin(Omega)*cos_inclination);
        ti[4] = -a0*(sin(omega)*sin(Omega) - cos(omega)*cos(Omega)*cos_inclination);
    }
    vector[T] anomaly;
    {
    	anomaly = true_anomaly_calc(T, t, P, t_peri);
    }
    array[2] vector[T] pos;
    {
    	pos = source_position(T, t, a, P, q, l, cos_inclination, Omega, omega, t_peri);
    }
    array[2] vector[T] pos2d;
     {
     	pos2d = projected_position_2d(T, t, a, P, cos_inclination, Omega, omega, t_peri);
     }
    array[2] vector[T] pos_tot;
    {
    	vector[T] diffs[2] = gaia_pos(T, ra_real, dec_real, plx_real, pmra_real, pmdec_real, gaia_mat);
    	pos_tot = system_pos(T, t, diffs[1], diffs[2], q, l, a, P, cos_inclination, Omega, omega, t_peri);
    }
    array[2] vector[T] pos_com;
    {
    	vector[T] diffs[2] = gaia_pos(T, ra_real, dec_real, plx_real, pmra_real, pmdec_real, gaia_mat);
    	pos_com[1] = diffs[1];
    	pos_com[2] = diffs[2];
    }
    array[2] real pos_init;
    {
    	pos_init = initial_position(a, P, q, l, cos_inclination, Omega, omega, t_peri);
    }
    vector[5] stan_obs;
    {
        vector[T] gaia_diffs[2] = gaia_pos(T, ra_real, dec_real, plx_real, pmra_real, pmdec_real, gaia_mat);
    	vector[T] total_diffs[2] = system_pos(T, t, gaia_diffs[1], gaia_diffs[2], q, l, a, P, cos_inclination, Omega, omega, t_peri);
        stan_obs = gaia_obs(T, sin_scan_angle, cos_scan_angle, total_diffs[1], total_diffs[2], inv_gaia);
    }
}"""
import stan
def build_model(gaia_params,gaia_errs,gaia_corr,g_mag,ruwe,ruwe_err,t_obs,scan_angle,plx_factor,ast_error,im_data,rv_data):
	global program_code
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
	A = design_matrix(t_obs-2016, scan_angle, plx_factor)
	C = np.linalg.solve(A.T @ A, np.eye(5))
	ACAT = A @ C @ A.T
	gaia_mat = gaia_matrix(t_obs-2016,ra,dec)
	gaia_mat_AL = gaia_matrix_AL(t_obs-2016,scan_angle,ra,dec)
	inv_gaia = gaia_inverse_AL(t_obs-2016,scan_angle,ra,dec)
	abs_mag = g_mag-5*np.log10(100./plx_obs)
	m1_mean = mass_from_mag(abs_mag)
	ast_noise = ast_error*np.random.randn(len(t_obs))
	rv_errs = 0.01*np.ones(rv_data.shape[0])
	data = {
		"ra": ra, 
		"dec": dec, 
		"pmra": pmra_obs, 
		"pmdec": pmdec_obs,
		"parallax": plx_obs,
		"gaia_mat":gaia_mat,
		"gaia_mat_AL":gaia_mat_AL,
		"inv_gaia":inv_gaia,
		"covar_matrix": cov_matrix,
		"ruwe_obs": ruwe,
		"T": t_obs.size,
		"t": t_obs-2016,
		"al_err": ast_error,
		"ACAT": ACAT.T,
		"scan_angle": scan_angle,
		"ruwe_err": ruwe_err,
		"ast_noise": ast_noise,
		"m1_mean": m1_mean,
		"g_mag": g_mag,
		"mag_err": 0.05,
		"rv_vals": rv_data[:,1],
		"rv_t": rv_data[:,0]-2016,
		"T2": rv_data.shape[0],
		"rv_errs": rv_errs,
	}
	model = stan.build(program_code, data=data)
	return model
