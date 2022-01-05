
import sys
import csv
import random as rd
import numpy as np
import math
import time

import sobol_seq

from scipy.integrate import quad
from scipy.special import i0
from scipy.stats import rice




## description of the arguments:
    ## param_sets : the parameter sets of the system parameters of our model.
    ## id_DR_HOR  : the indicator of the metrics [DR/HOR, DR -> the data rate, HOR -> the handover rate].
    ## flag_sobol : the flag to use the sobol sequences for monte-calro integration calculus [True/False].
    ## McNum_sobol: the sample number of the monte-calro integration calculus using the sobol sequences [value type: int].
    ## DivNum_HOR : the division number of the trapezoidal integration calculus for HOR [value type: int].
    ## DivNum_DR1 : the division number of the trapezoidal integration calculus for DR (first kind) [value type: int].
    ## DivNum_DR2 : the division number of the trapezoidal integration calculus for DR (second kind) [value type: int].
    ## DivNum_DR3 : the division number of the trapezoidal integration calculus for DR (third kind) [value type: int].


def out__DR_HOR(param_sets, id_DR_HOR, flag_sobol, McNum_sobol=1024, DivNum_HOR=100, DivNum_DR1=400, DivNum_DR2=20, DivNum_DR3=4):

    ### parameter input
    s1, s2, v, lam1, lam2p, mb, sigma, P0, P1, beta = param_sets


    ## random number generater for montecalro integration.
    ##   - lower bounds of integration variables must be 0
    ##   - uppber bounds are callable with the list: UB_l
    def rd_gen(UB_l, samplenum, flag_sobol=False):    ## UB_l: list of upper bounds of integration variable
        if not type(samplenum) == type(1):
            print('Error(rd_gen): argument samplenum must be Integer type.')
            sys.exit()
        if not type(UB_l) == type([1.0, 2.0, 3.0]):
            print('Error(rd_gen): argument UB_l must be List type.')
            sys.exit()
        varnum = len(UB_l)    ## number of integration variables
        if flag_sobol:
            return sobol_seq.i4_sobol_generate(varnum, samplenum) * np.array(UB_l)
        else:
            #np.random.seed(0)
            return np.random.uniform(0, 1, varnum*samplenum).reshape(samplenum, varnum) * np.array(UB_l)



    def get_Pbij(Pi, Pj, beta):
        return [ [1.0, (Pi/Pj)**(1/beta)], [(Pj/Pi)**(1/beta), 1.0] ]
    def Wur(u, r, theta):
        return np.sqrt(u**2 + r**2 - 2*u*r*np.cos(theta))
    
    def dens_d(x, z):
        return rice.pdf(x, z/sigma, scale=sigma)
    def F_d(r, z): 
        return rice.cdf(r, z/sigma, scale=sigma)
    
    def I00(r, z):
        return np.exp( -mb * F_d(Pb[0][0]*r, z) )
    def I01(r, z):
        return np.exp( -mb * F_d(Pb[0][1]*r, z) )
    def I10(r, z):
        return np.exp( -mb * F_d(Pb[1][0]*r, z) )
    def I11(r, z):
        return np.exp( -mb * F_d(Pb[1][1]*r, z) )


    #### out_HOR starts: the function for caluculating HOR (the handover rate) ####
    def out_HOR():
        def B1(u, r, theta):
            def func_phi(u, r, bx):
                return np.arccos( np.clip((u-bx)/np.sqrt(r**2+u**2-2*u*bx), -1, 1) )
            bx = r*np.cos(theta)
            by = r*np.sin(theta)
            return (np.pi-func_phi(u, r, bx)-theta)*r**2 + ( by-2*(np.pi-func_phi(u, r, bx))*bx )*u + (np.pi-func_phi(u, r, bx))*u**2
        def B2(u, r, theta):
            if u > 2*r*np.cos(theta):
                return np.pi * Pb[0][1]**2 * (Wur(u,r,theta)**2 - r**2)
            else:
                return 0.0
        def J1(r, theta, z): 
            if s1*v > Pb[1][0]*r:
                return np.exp( -mb * F_d(Pb[1][0]*Wur(s1*v,r,theta), z) )
            else:
                return 1.0
        def J2(r, theta, z):
            if s2*v > Pb[1][1]*r:
                return np.exp( -mb * F_d(Pb[1][1]*Wur(s2*v,r,theta), z) )
            else:
                return 1.0



        def integ__H1(r):

            def integ__H1_1(r):
                def trapz__H1_1(r):
                    def func__H1_1(r, w):
                        return ( dens_d(r, w) * I11(r, w)  )*w*(1+w**2)
                    ### Trapezoidal integral over [0, 2*np.pi]
                    ## w = tan(psi)
                    w_UB = r + 0.5; w_LB = np.max([0.0, r - 0.5]);    ## the margin 0.5 could be managed depending on sigma: the daughter variance.
                    psi_range = np.linspace(np.arctan(w_LB), np.arctan(w_UB), DivNum_HOR+1)
                    return np.nanmean([func__H1_1(r, np.tan(psi)) for psi in psi_range])*(np.arctan(w_UB) - np.arctan(w_LB))
                return np.exp(-np.pi*lam1*Pb[0][1]**2*r**2) * trapz__H1_1(r)
            def integ__H1_2(r):
                def func__H1_2(r, w):
                    return (1 - I11(r, w))*w*(1+w**2)
                ### Trapezoidal integral over [0, 2*np.pi]
                ## w = tan(psi)
                psi_range = np.linspace(0, np.pi/2, DivNum_HOR+1)
                trapz__H1_2 = np.nanmean([func__H1_2(r, np.tan(psi)) for psi in psi_range])*np.pi/2
                return np.exp( -2*np.pi*lam2p*trapz__H1_2 )

            return integ__H1_1(r) * integ__H1_2(r) * (1+r**2)
        

        def integ__H2(r, theta):

            def integ__H2_1(r, theta):
                return np.exp(-lam1*(np.pi*r**2 + B1(s1*v, r, theta)))
            def integ__H2_2(r, theta):
                def func__H2_2(r, theta, w):
                    return (2 - I10(r, w) - J1(r, theta, w))*w*(1+w**2)
                ### Trapezoidal integral over [0, 2*np.pi]
                ## w = tan(psi)
                psi_range = np.linspace(0, np.pi/2, DivNum_HOR+1)
                trapz__H2_2 = np.nanmean([func__H2_2(r, theta, np.tan(psi)) for psi in psi_range])*np.pi/2
                return np.exp( -2*np.pi*lam2p*trapz__H2_2 )

            return integ__H2_1(r, theta) * integ__H2_2(r, theta) * r*(1+r**2)


        def integ__H3(r, theta):

            def integ__H3_1(r, theta):
                def trapz__H3_1(r):
                    def func__H3_1(r, w):
                        return ( dens_d(r, w) * I11(r, w)  )*w*(1+w**2)
                    ### Trapezoidal integral over [0, 2*np.pi]
                    ## w = tan(psi)
                    w_UB = r + 0.5; w_LB = np.max([0.0, r - 0.5]);    ## the margin 0.5 could be managed depending on sigma: the daughter variance.
                    psi_range = np.linspace(np.arctan(w_LB), np.arctan(w_UB), DivNum_HOR+1)
                    return np.nanmean([func__H3_1(r, np.tan(psi)) for psi in psi_range])*(np.arctan(w_UB) - np.arctan(w_LB))
                return np.exp(-lam1*(np.pi*Pb[0][1]**2*r**2 + B2(s2*v, r, theta))) * trapz__H3_1(r)
            def integ__H3_2(r, theta):
                def func__H3_2(r, theta, w):
                    return (2 - I11(r, w) - J2(r, theta, w))*w*(1+w**2)
                ## Trapezoidal integral over [0, 2*np.pi]
                ## w = tan(psi)
                psi_range = np.linspace(0, np.pi/2, DivNum_HOR+1)
                trapz__H3_2 = np.nanmean([func__H3_2(r, theta, np.tan(psi)) for psi in psi_range])*np.pi/2
                return np.exp( -2*np.pi*lam2p*trapz__H3_2 )

            return integ__H3_1(r, theta) * integ__H3_2(r, theta) * (1+r**2)


        ### montecalro integration
        UB_phi, UB_vphi, UB_theta, UB2_theta = np.pi/2, np.pi/2, 2*np.pi, np.pi
        UB_phi_alt = np.pi/4


        mcint__H1 = UB_phi_alt * np.nanmean([integ__H1(np.tan(phi)) for phi in rd_gen([UB_phi_alt], McNum_sobol, flag_sobol)])
        mcint__H2 = UB_phi*UB2_theta * np.nanmean([integ__H2(np.tan(phi), theta) for (phi, theta) in rd_gen([UB_phi, UB2_theta], McNum_sobol, flag_sobol)])
        mcint__H3 = UB_phi_alt*UB_theta * np.nanmean([integ__H3(np.tan(phi), theta) for (phi, theta) in rd_gen([UB_phi_alt, UB_theta], McNum_sobol, flag_sobol)])

        return 1/s1 + 2*np.pi*mb*lam2p*(1/s2 - 1/s1)*mcint__H1 - 2*lam1/s1*mcint__H2 - mb*lam2p/s2*mcint__H3
    #### out_HOR ends: the function for calculating HOR (the handover rate) ####


    #### out_DR starts: the function for calculating DR (the data rate) ####
    def out_DR():


        def K_beta(beta):
            return np.pi**2/beta/np.sin(2*np.pi/beta)

        def vphi_ti(u, r, x):
            if not u:
                u = 0.0000001
            if not x:
                x = 0.0000001
            return np.arccos(np.clip((u**2+x**2-r**2)/(2*u*x),-1,1))

        def Rho1(y, r, theta, t):
            def integrand_rho(x, y, r, theta, t):
                if not y or not r or not t:
                    frcinv = 0.0
                else:
                    frcinv = ( 1 + x**beta/(Pb[0][0]**beta*Wur(t*v,r,theta)**beta*y) )**(-1)
                return vphi_ti(t*v, Pb[0][0]*r, x) * frcinv * x

            epnt_part1 =  K_beta(beta)*Pb[0][0]**2*Wur(t*v,r,theta)**2*y**(2/beta) 
            epnt_part2 =  quad(integrand_rho, 0, t*v + Pb[0][0]*r, args=(y,r,theta,t))[0]
            return np.exp( -2*lam1*(epnt_part1 - epnt_part2) )
        def Rho2(y, r, theta, t):
            def integrand_rho(x, y, r, theta, t):
                if not y or not r or not t:
                    frcinv = 0.0
                else:
                    frcinv = ( 1 + x**beta/(Pb[0][1]**beta*Wur(t*v,r,theta)**beta*y) )**(-1)
                return vphi_ti(t*v, Pb[0][1]*r, x) * frcinv * x

            epnt_part1 =  K_beta(beta)*Pb[0][1]**2*Wur(t*v,r,theta)**2*y**(2/beta) 
            epnt_part2 =  quad(integrand_rho, 0, t*v + Pb[0][1]*r, args=(y,r,theta,t))[0]
            return np.exp( -2*lam1*(epnt_part1 - epnt_part2) )

        def K1(y, r, theta, z, t):
            def trapz__K1(y, r, theta, z, t):
                def func__K1(x, y, r, theta, z, t):
                    if not y or not r or not t:
                        frcinv = 0.0
                    else:
                        frcinv = ( 1 + x**beta/(Pb[1][0]**beta * Wur(t*v, r, theta)**beta * y) )**(-1)
                    return dens_d(x, z) * frcinv * (1+x**2)
                ### Trapezoidal integral over [x_LB, x_UB]
                ## x = tan(vpsi)
                x_UB = z + 0.5; x_LB = np.max([0.0, z - 0.5]);    ## the margin 0.5 could be managed depending on sigma: the daughter variance.
                vpsi_range = np.linspace(np.arctan(x_LB), np.arctan(x_UB), DivNum_DR2+1); vpsi_max, vpsi_min = vpsi_range[-1], vpsi_range[0]
                ex_term = (vpsi_max - vpsi_min)/(2*DivNum_DR2)*( func__K1(np.tan(vpsi_min),y,r,theta,z,t) - func__K1(np.tan(vpsi_max),y,r,theta,z,t) )
                return np.nanmean([ func__K1(np.tan(vpsi), y, r, theta, z, t) for vpsi in vpsi_range[1:] ])*(np.arctan(x_UB) - np.arctan(x_LB)) + ex_term
            return np.exp( -mb * trapz__K1(y, r, theta, z, t) )
        def K2(y, r, theta, z, t):
            def trapz__K2(y, r, theta, z, t):
                def func__K2(x, y, r, theta, z, t):
                    if not y or not r or not t:
                        frcinv = 0.0
                    else:
                        frcinv = ( 1 + x**beta/(Pb[1][1]**beta * Wur(t*v, r, theta)**beta * y) )**(-1)
                    return dens_d(x, z) * frcinv * (1+x**2)
                ### Trapezoidal integral over [x_LB, x_UB]
                ## x = tan(vpsi)
                x_UB = z + 0.5; x_LB = np.max([0.0, z - 0.5]);    ## the margin 0.5 could be managed depending on sigma: the daughter variance.
                vpsi_range = np.linspace(np.arctan(x_LB), np.arctan(x_UB), DivNum_DR2+1); vpsi_max, vpsi_min = vpsi_range[-1], vpsi_range[0]
                ex_term = (vpsi_max - vpsi_min)/(2*DivNum_DR2)*( func__K2(np.tan(vpsi_min),y,r,theta,z,t) - func__K2(np.tan(vpsi_max),y,r,theta,z,t) )
                return np.nanmean([ func__K2(np.tan(vpsi), y, r, theta, z, t) for vpsi in vpsi_range[1:] ])*(np.arctan(x_UB) - np.arctan(x_LB)) + ex_term
            return np.exp( -mb * trapz__K2(y, r, theta, z, t) )



        def integ__D1(y, r, theta, t):

            def integ__D1_1(y, r, theta, t):
                return np.exp( -lam1*np.pi*r**2 - sigmaN**2*Wur(t*v,r,theta)**beta*y/P0 ) / (1 + y) * Rho1(y, r, theta, t)
            def integ__D1_2(y, r, theta, t):
                def trapz__D1_2(y, r, theta, t):
                    def func__D1_2(y, r, theta, w, t):
                        return ( 2 - I10(r, w) - K1(y, r, theta, w, t) )*w*(1+w**2)
                    ### Trapezoidal integral over [0, np.pi/2]
                    ## w = tan(psi)
                    psi_range = np.linspace(0, np.pi/2, DivNum_DR2+1); psi_max, psi_min = psi_range[-1], psi_range[0]
                    ex_term = (psi_max - psi_min)/(2*DivNum_DR2)*( func__D1_2(y,r,theta,np.tan(psi_min),t) - func__D1_2(y,r,theta,np.tan(psi_max),t) )
                    return np.nanmean([ func__D1_2(y, r, theta, np.tan(psi), t) for psi in psi_range[1:] ])*np.pi/2 + ex_term
                return np.exp( -2*np.pi*lam2p*trapz__D1_2(y, r, theta, t) )

            return integ__D1_1(y, r, theta, t) * integ__D1_2(y, r, theta, t) * (1+y**2) * r*(1+r**2) 

        def integ__D2(y, r, theta, t):
            def integ__D2_1(y, r, theta, t):
                def trapz__D2_1(r):
                    def func__D2_1(r, w):
                        return ( dens_d(r, w) * I11(r, w)  )*w*(1+w**2)
                    ### Trapezoidal integral over [w_LB, w_UB]
                    ## w = tan(psi)
                    w_UB = r + 0.5; w_LB = np.max([0.0, r - 0.5]);    ## the margin 0.5 could be managed depending on sigma: the daughter variance.
                    psi_range = np.linspace(np.arctan(w_LB), np.arctan(w_UB), DivNum_DR2+1); psi_max, psi_min = psi_range[-1], psi_range[0]
                    ex_term = (psi_max - psi_min)/(2*DivNum_DR2)*( func__D2_1(r,np.tan(psi_min)) - func__D2_1(r,np.tan(psi_max)) )
                    return np.nanmean([ func__D2_1(r, np.tan(psi)) for psi in psi_range[1:] ])*(np.arctan(w_UB) - np.arctan(w_LB)) + ex_term
                return np.exp( -lam1*np.pi*Pb[0][1]**2*r**2 - sigmaN**2*Wur(t*v,r,theta)**beta*y/P1 ) / (1 + y) * Rho2(y, r, theta, t) * trapz__D2_1(r)
            def integ__D2_2(y, r, theta, t):
                def trapz__D2_2(y, r, theta, t):
                    def func__D2_2(y, r, theta, w, t):
                        return ( 2 - I11(r, w) - K2(y, r, theta, w, t) )*w*(1+w**2)
                    ## Trapezoidal integral over [0, np.pi/2]
                    ## w = tan(psi)
                    psi_range = np.linspace(0, np.pi/2, DivNum_DR2+1); psi_max, psi_min = psi_range[-1], psi_range[0]
                    ex_term = (psi_max - psi_min)/(2*DivNum_DR2)*( func__D2_2(y,r,theta,np.tan(psi_min),t) - func__D2_2(y,r,theta,np.tan(psi_max),t) )
                    return np.nanmean([ func__D2_2(y, r, theta, np.tan(psi), t) for psi in psi_range[1:] ])*np.pi/2 + ex_term
                return np.exp( -2*np.pi*lam2p*trapz__D2_2(y, r, theta, t) )

            return integ__D2_1(y, r, theta, t) * integ__D2_2(y, r, theta, t) * (1+y**2) * (1+r**2)


        ### montecalro integration
        UB_eps, UB_phi, UB_theta = np.pi/2, np.pi/2, np.pi
        UB_phi_alt = np.pi/4


        ### Variable changing for y and r:
            ## y = tan(eps)
            ## r = tan(phi)

        def mcint__D1(t):
            def trapzint_phi__D1(t):
                def trapzint_eps__D1(r, t):
                    def trapzint_theta__D1(y, r, t):
                        ## Trapezoidal integral over [0, UB_theta]
                        theta_range = np.linspace(0, UB_theta, DivNum_DR3+1); theta_max, theta_min = theta_range[-1], theta_range[0]
                        exterm_theta__D1 = (theta_max - theta_min)/(2*DivNum_DR3)*( integ__D1(y,r,theta_min,t) - integ__D1(y,r,theta_max,t) )
                        return np.nanmean([ integ__D1(y,r,theta,t) for theta in theta_range[1:] ])*UB_theta + exterm_theta__D1
                    ### Trapezoidal integral over [0, UB_eps]
                    ## y = tan(eps)
                    eps_range = np.linspace(0, UB_eps, DivNum_DR1+1); eps_max, eps_min = eps_range[-1], eps_range[0]
                    exterm_eps__D1 = (eps_max - eps_min)/(2*DivNum_DR1)*( trapzint_theta__D1(np.tan(eps_min),r,t) - trapzint_theta__D1(np.tan(eps_max),r,t) )
                    return np.nanmean([ trapzint_theta__D1(np.tan(eps),r,t) for eps in eps_range[1:] ])*UB_eps + exterm_eps__D1
                ## Trapezoidal integral over [0, UB_phi]
                ## r = tan(phi)
                phi_range = np.linspace(0, UB_phi, DivNum_DR1+1); phi_max, phi_min = phi_range[-1], phi_range[0]
                exterm_phi__D1 = (phi_max - phi_min)/(2*DivNum_DR1)*( trapzint_eps__D1(np.tan(phi_min),t) - trapzint_eps__D1(np.tan(phi_max),t) )
                return np.nanmean([ trapzint_eps__D1(np.tan(phi),t) for phi in phi_range[1:] ])*UB_phi + exterm_phi__D1
            return 2*lam1 * trapzint_phi__D1(t)

        def mcint__D2(t):
            def trapzint_phi__D2(t):
                def trapzint_eps__D2(r, t):
                    def trapzint_theta__D2(y, r, t):
                        ### Trapezoidal integral over [0, UB_theta]
                        theta_range = np.linspace(0, UB_theta, DivNum_DR3+1); theta_max, theta_min = theta_range[-1], theta_range[0]
                        exterm_theta__D2 = (theta_max - theta_min)/(2*DivNum_DR3)*( integ__D2(y,r,theta_min,t) - integ__D2(y,r,theta_max,t) )
                        return np.nanmean([ integ__D2(y,r,theta,t) for theta in theta_range[1:] ])*UB_theta + exterm_theta__D2
                    ### Trapezoidal integral over [0, UB_eps]
                    ## y = tan(eps)
                    eps_range = np.linspace(0, UB_eps, DivNum_DR1+1); eps_max, eps_min = eps_range[-1], eps_range[0]
                    exterm_eps__D2 = (eps_max - eps_min)/(2*DivNum_DR1)*( trapzint_theta__D2(np.tan(eps_min),r,t) - trapzint_theta__D2(np.tan(eps_max),r,t) )
                    return np.nanmean([ trapzint_theta__D2(np.tan(eps),r,t) for eps in eps_range[1:] ])*UB_eps + exterm_eps__D2
                ### Trapezoidal integral over [0, UB_phi_alt]
                ## r = tan(phi)
                phi_range = np.linspace(0, UB_phi_alt, DivNum_DR1+1); phi_max, phi_min = phi_range[-1], phi_range[0]
                exterm_phi__D2 = (phi_max - phi_min)/(2*DivNum_DR1)*( trapzint_eps__D2(np.tan(phi_min),t) - trapzint_eps__D2(np.tan(phi_max),t) )
                return np.nanmean([ trapzint_eps__D2(np.tan(phi),t) for phi in phi_range[1:] ])*UB_phi_alt + exterm_phi__D2
            return 2*mb*lam2p * trapzint_phi__D2(t)


        ### make mcD1_L,mcD2_L
        mcD1_L, mcD2_L = [], []
        for t1 in np.arange(s1):
            start = time.time()
            result_D1_val = mcint__D1(t1)
            mcD1_L.append(result_D1_val)
            finish = time.time()
            print('t1 = {}, result_DR(term1) = {}, elapsed time = {}'.format(t1, result_D1_val, finish - start))
        for t2 in np.arange(s2):
            start = time.time()
            result_D2_val = mcint__D2(t2)
            mcD2_L.append(result_D2_val)
            finish = time.time()
            print('t2 = {}, result_DR(term2) = {}, elapsed time = {}'.format(t2, result_D2_val, finish - start))

        ### make DR1_L,DR2_L: time average of mcint__D1,mcint__D2 
        DR1_L = ['']*len(mcD1_L); DR2_L = ['']*len(mcD2_L)
        for i in range(len(mcD1_L)):
            DR1_L[i] = np.mean(mcD1_L[:i+1])
        for j in range(len(mcD2_L)):
            DR2_L[j] = np.mean(mcD2_L[:j+1])

        ### make outout matrix
        ## header: <1> skipping time1(t1), <2> skipping time2(t2), <3> result value (DR1+DR2)
        output_matrix = []
        for i in range(len(DR1_L)):
            t1 = np.arange(s1)[i]
            for j in range(len(DR2_L)):
                t2 = np.arange(s2)[j]
                if not t1 < t2:
                    output_matrix.append([t1 + 1.0, t2 + 1.0, DR1_L[i], DR2_L[j], DR1_L[i] + DR2_L[j] ])
                else:
                    output_matrix.append([t1 + 1.0, t2 + 1.0, '-'])

        return output_matrix
    #### out_DR ends: the function for calculating DR (the data rate) ####



    #### the function calls start ####
    sigmaN = 0.0    ## sigmaN: noise power (omitted in this computation).

    ## index 0: ppp index    index 1: tcp index
    Pb = get_Pbij(P0, P1, beta)

    
    if id_DR_HOR == 'DR':
        return out_DR()
    elif id_DR_HOR == 'HOR':
        start = time.time()
        result_HOR = out_HOR()
        finish = time.time()
        print('result_HOR = {}, elapsed time = {}'.format(result_HOR, finish - start))
        return [result_HOR, finish - start]    ## return a line of row
    else:
        print('Error (out__DR_HOR): the 2rd argument id_DR_HOR must be "DR" or "HOR".')
        return [None, None]
    #### the function calls end ####





    
#### main function ####

def NoticeMessages():
    print('<< Notice >>')
    print('11 arguments are required to be written in the input csv.')
    print('the 11 args -> 1 : skipping time for ppp')
    print('               2 : skipping time for tcp')
    print('               3 : moving velocity of a user')
    print('               4 : intensity for ppp')
    print('               5 : parent intensity for tcp')
    print('               6 : daughter intensity for tcp')
    print('               7 : daighter variance for tcp')
    print('               8 : transmitting power for ppp (macro base stations)')
    print('               9 : transmitting power for tcp (small base stations)')
    print('               10: path-loss exponent')
    print('               11: curve type <<must be either of "straight"/"circle"/"spiral">>  ')

if __name__ == "__main__":
    argvs = sys.argv

    if not len(argvs[1:]) == 2:
        print('InputError: 2 arguments need to be input.')
        print('                1st: csv filename')
        print('                2nd: metrics type (DR/HOR)')
        sys.exit()

    InputCsvFilename = argvs[1]
    if not InputCsvFilename[-4:] == '.csv':
        print('InputError: Please set a csv file (containing 11 parameters) for the 1st argument.')
        NoticeMessages()
        sys.exit()
    else:
        f = open(InputCsvFilename, 'r')
        csvreader = csv.reader(f)
        header = next(csvreader)
        matrix = [v for v in csvreader]
        f.close()

    MetricsType = argvs[2].lower()
    if MetricsType not in ['dr', 'hor']:
        print('InputError: Please set either of DR/HOR (metrics type) for the 2nd argument.')
        sys.exit()


    def outputcsvs_from_inputcsv(matrix):
        def make_parameters_set(matrix_row):
            error_flag = False
            try:
                paras10_L = list(map(float, matrix_row[:10]))
            except ValueError:
                print('\nInput Error: 11 arguments are required to be written in each row of the input csv.')
                print('skip message: the parameter set of the {}th row is omitted due to missing the requirements...\n'.format(i+1))
                paras10_L = None; error_flag = True

            if matrix_row[10].lower() in ['straight', 'circle', 'spiral']:
                curve_type = matrix_row[10].lower()
            else:
                print('\nInput Error: the 11th argument must be either of "straight"/"circle"/"spiral" .')
                print('skip message: the parameter set of the {}th row is omitted due to missing the requirements...\n'.format(i+1))
                curve_type = None; error_flag = True
            return paras10_L, curve_type, error_flag


        if MetricsType == 'hor':
            result_header = ['handover_rate', 'elapsed_time']
            result_rows_L = []
            NoticeMessagesFlag = False
            for i in range(len(matrix)):
                ParamSets, CurveType, ErrorFlag = make_parameters_set(matrix[i])

                if not ErrorFlag:
                    result_row = out__DR_HOR(ParamSets, 'HOR', flag_sobol=True)
                else:
                    NoticeMessagesFlag = True
                    result_row = ['', '']
                    continue
                result_rows_L.append(result_row)
            if NoticeMessagesFlag:
                NoticeMessages()
        else:    ## if MetricsType == 'dr':
            result_header = ['skipping time1 (macro)', 'skipping time2 (small)', 'data_rate (macro)', 'data_rate (small)', 'data_rate (total)']
            ParamSets, CurveType, ErrorFlag = make_parameters_set(matrix[-1])    ## input the last row of the matrix

            if not ErrorFlag:
                result_rows_L = out__DR_HOR(ParamSets, 'DR', flag_sobol=True)
            else:
                result_rows_L = [['', '']]
                NoticeMessages()


        OutputCsvFilename = 'result' + MetricsType.upper() + '_' + InputCsvFilename
        f = open(OutputCsvFilename, 'w')
        csvwriter = csv.writer(f)
        csvwriter.writerow(result_header)
        csvwriter.writerows(result_rows_L)
        f.close()
            
    argvs = sys.argv

    if not argvs[1][-4:] == '.csv':
        print('Input Error:  the input argument must be a csv file containing 11 kinds of parameters.')
        NoticeMessages()
        sys.exit()
    else:
        pass


    f = open(argvs[1], 'r')
    csvreader = csv.reader(f)
    header = next(csvreader)
    matrix = [v for v in csvreader]
    f.close()

    outputcsvs_from_inputcsv(matrix)

