
import sys
import csv
import random as rand
import numpy as np
import math
import time

from scipy import integrate
from scipy import stats
import sobol_seq



## description of the arguments:
    ## param_sets : the parameter sets of the system parameters of our model.
        ## s1/s2    : the skipping time for the 1st/2nd (macro/small) tier.
        ## veloc    : the velocity of the moving UE.
        ## lam1     : the intensity of the PPP in the 1st (macro) tier.
        ## lam2p/mb : the intensity of the parent/daughter points of the PCP in the 2nd (small) tier.
        ## sigma/rd : the clustering variance/the cluster radius for the TCP/MCP in the 2nd (small) tier.
        ## P1/P2    : the transmission power in the 1st/2nd (macro/small) tier.
        ## beta     : the common path-loss exponent in the 1st/2nd (macro/small) tier.
    ## PCP_id  : the indicator of the specific PCP [TCP/MCP, TCP -> Thomas Cluster Process, MCP -> Matern Cluster Process].
    ## McNum: the sample number of the monte-calro integration calculus using the sobol sequences [value type: int].
    ## DivNum : the division number of the trapezoidal integration calculus for HOR [value type: int].


def output_HOR(param_sets, PCP_id):

    ### parameter input
    if PCP_id == 'TCP':
        s1, s2, veloc, lam1, lam2p, mb, sigma, P1, P2, beta = param_sets
    elif PCP_id == 'MCP':
        s1, s2, veloc, lam1, lam2p, mb, rd, P1, P2, beta = param_sets
    else:
        print('Error, output_DR: Please input a valid PCP_id; "TCP" or "MCP". ')
        return None

    McNum=1024
    DivNum=100

    ## random number generater for montecalro integration.
    ##   - lower bounds of integration variables must be 0
    ##   - uppber bounds are callable with the list: UB_l
    def rd_gen(UB_l, samplenum):    ## UB_l: list of upper bounds of integration variable
        if not type(samplenum) == type(1):
            print('Error(rd_gen): argument samplenum must be Integer type.')
            sys.exit()
        if not type(UB_l) == type([1.0, 2.0, 3.0]):
            print('Error(rd_gen): argument UB_l must be List type.')
            sys.exit()
        varnum = len(UB_l)    ## number of integration variables
        if varnum > 1:
            return sobol_seq.i4_sobol_generate(varnum, samplenum) * np.array(UB_l)
        else:    ## the case len(UB_L) == 1.
            return sobol_seq.i4_sobol_generate(varnum, samplenum).flatten() * UB_l[0]

    def P_bar(tier_i, tier_j):
        def P(tier_i):
            if tier_i == 1:
                return P1
            elif tier_i == 2:
                return P2
            else:
                print('Error;def P(tier_i): tier_i must be either 1 or 2.')
                sys.exit()
        return (P(tier_j)/P(tier_i))**(1/beta)
    def wcos(r, u, theta):
        return np.sqrt(r**2 + u**2 - 2*r*u*np.cos(theta))
    
    def func_I11(r, z):
        return np.exp( -mb * func_Fd(P_bar(1, 1)*r, z) )
    def func_I21(r, z):
        return np.exp( -mb * func_Fd(P_bar(2, 1)*r, z) )
    def func_I12(r, z):
        return np.exp( -mb * func_Fd(P_bar(1, 2)*r, z) )
    def func_I22(r, z):
        return np.exp( -mb * func_Fd(P_bar(2, 2)*r, z) )

    ### Specific functions for the TCP/MCP.
    def func_fd(r, z):
        if PCP_id == 'TCP':
            return stats.rice.pdf(r, z/sigma, scale=sigma)
        elif PCP_id == 'MCP':
            if r<=max(rd - z, 0):
                return 2*r/rd**2
            elif abs(rd - z)<=r and r<=rd + z: 
                return 1/np.pi*np.arccos( round((r**2 + z**2 - rd**2)/(2*r*z), 8) )*2*r/rd**2
            else:
                return 0.0
        else:
            print('Error, output_DR: Please input a valid PCP_id; "TCP" or "MCP". ')
            return None
    def func_Fd(r, z): 
        if PCP_id == 'TCP':
            return stats.rice.cdf(r, z/sigma, scale=sigma)
        elif PCP_id == 'MCP':
            def integd_x(L_x, z):
                return np.array([ x*np.arccos( round((x**2 + z**2 - rd**2)/(2*x*z), 8) ) for x in L_x])

            if min(r, abs(rd - z)) == min(r, rd + z):
                return (min(r, max(rd - z, 0))**2)/rd**2
            else:
                L_x = np.linspace(min(r, abs(rd - z)), min(r, rd + z), DivNum+1)
                return (min(r, max(rd - z, 0))**2 + 2/np.pi*integrate.trapz(integd_x(L_x, z), L_x))/rd**2
    

    #### out_HOR starts: the function for caluculating HOR (the handover rate) ####
    def func_B1(u, r, theta):
        def func_phi(u, r, bx):
            return np.arccos( np.clip((u-bx)/np.sqrt(r**2+u**2-2*u*bx), -1, 1) )
        bx = r*np.cos(theta)
        by = r*np.sin(theta)
        return (np.pi-func_phi(u, r, bx)-theta)*r**2 + ( by-2*(np.pi-func_phi(u, r, bx))*bx )*u + (np.pi-func_phi(u, r, bx))*u**2
    def func_B2(u, r, theta):
        if u > 2*r*np.cos(theta):
            return np.pi * P_bar(2, 1)**2 * (wcos(r,u,theta)**2 - r**2)
        else:
            return 0.0
    def func_J1(r, theta, z): 
        if s1*veloc > P_bar(1, 2)*r:
            return np.exp( -mb * func_Fd(P_bar(1, 2)*wcos(r,s1*veloc,theta), z) )
        else:
            return 1.0
    def func_J2(r, theta, z):
        if s2*veloc > P_bar(2, 2)*r:
            return np.exp( -mb * func_Fd(P_bar(2, 2)*wcos(r,s2*veloc,theta), z) )
        else:
            return 1.0



    def integ__H1(r):

        def integ__H1_1(r):
            def trapz__H1_1(r):
                def func__H1_1(r, w):
                    return ( func_fd(r, w) * func_I22(r, w)  )*w*(1+w**2)
                ### Trapezoidal integral over [0, 2*np.pi]
                ## w = tan(psi)
                w_UB = r + 0.5; w_LB = np.max([0.0, r - 0.5]);    ## the margin 0.5 could be managed depending on sigma: the daughter variance.
                psi_range = np.linspace(np.arctan(w_LB), np.arctan(w_UB), DivNum+1)
                return np.nanmean([func__H1_1(r, np.tan(psi)) for psi in psi_range])*(np.arctan(w_UB) - np.arctan(w_LB))
            return np.exp(-np.pi*lam1*P_bar(2, 1)**2*r**2) * trapz__H1_1(r)
        def integ__H1_2(r):
            def func__H1_2(r, w):
                return (1 - func_I22(r, w))*w*(1+w**2)
            ### Trapezoidal integral over [0, 2*np.pi]
            ## w = tan(psi)
            psi_range = np.linspace(0, np.pi/2, DivNum+1)
            trapz__H1_2 = np.nanmean([func__H1_2(r, np.tan(psi)) for psi in psi_range])*np.pi/2
            return np.exp( -2*np.pi*lam2p*trapz__H1_2 )

        return integ__H1_1(r) * integ__H1_2(r) * (1+r**2)
    

    def integ__H2(r, theta):

        def integ__H2_1(r, theta):
            return np.exp(-lam1*(np.pi*r**2 + func_B1(s1*veloc, r, theta)))
        def integ__H2_2(r, theta):
            def func__H2_2(r, theta, w):
                return (2 - func_I12(r, w) - func_J1(r, theta, w))*w*(1+w**2)
            ### Trapezoidal integral over [0, 2*np.pi]
            ## w = tan(psi)
            psi_range = np.linspace(0, np.pi/2, DivNum+1)
            trapz__H2_2 = np.nanmean([func__H2_2(r, theta, np.tan(psi)) for psi in psi_range])*np.pi/2
            return np.exp( -2*np.pi*lam2p*trapz__H2_2 )

        return integ__H2_1(r, theta) * integ__H2_2(r, theta) * r*(1+r**2)


    def integ__H3(r, theta):

        def integ__H3_1(r, theta):
            def trapz__H3_1(r):
                def func__H3_1(r, w):
                    return ( func_fd(r, w) * func_I22(r, w)  )*w*(1+w**2)
                ### Trapezoidal integral over [0, 2*np.pi]
                ## w = tan(psi)
                w_UB = r + 0.5; w_LB = np.max([0.0, r - 0.5]);    ## the margin 0.5 could be managed depending on sigma: the daughter variance.
                psi_range = np.linspace(np.arctan(w_LB), np.arctan(w_UB), DivNum+1)
                return np.nanmean([func__H3_1(r, np.tan(psi)) for psi in psi_range])*(np.arctan(w_UB) - np.arctan(w_LB))
            return np.exp(-lam1*(np.pi*P_bar(2, 1)**2*r**2 + func_B2(s2*veloc, r, theta))) * trapz__H3_1(r)
        def integ__H3_2(r, theta):
            def func__H3_2(r, theta, w):
                return (2 - func_I22(r, w) - func_J2(r, theta, w))*w*(1+w**2)
            ## Trapezoidal integral over [0, 2*np.pi]
            ## w = tan(psi)
            psi_range = np.linspace(0, np.pi/2, DivNum+1)
            trapz__H3_2 = np.nanmean([func__H3_2(r, theta, np.tan(psi)) for psi in psi_range])*np.pi/2
            return np.exp( -2*np.pi*lam2p*trapz__H3_2 )

        return integ__H3_1(r, theta) * integ__H3_2(r, theta) * (1+r**2)


    ### montecalro integration
    UB_phi, UB_vphi, UB_theta, Ufunc_B2_theta = np.pi/2, np.pi/2, 2*np.pi, np.pi
    UB_phi_alt = np.pi/4

    mcint__H1 = UB_phi_alt * np.nanmean([integ__H1(np.tan(phi)) for phi in rd_gen([UB_phi_alt], McNum)])
    mcint__H2 = UB_phi*Ufunc_B2_theta * np.nanmean([integ__H2(np.tan(phi), theta) for (phi, theta) in rd_gen([UB_phi, Ufunc_B2_theta], McNum)])
    mcint__H3 = UB_phi_alt*UB_theta * np.nanmean([integ__H3(np.tan(phi), theta) for (phi, theta) in rd_gen([UB_phi_alt, UB_theta], McNum)])


    #return 1/s1 + 2*np.pi*mb*lam2p*(1/s2 - 1/s1)*mcint__H1 - 2*lam1/s1*mcint__H2 - mb*lam2p/s2*mcint__H3
    if s1 and s2:
        start = time.time()
        HORt1 = 1/s1 - 2*np.pi*mb*lam2p*1/s1*mcint__H1 - 2*lam1/s1*mcint__H2
        end = time.time()
        etime1 = end - start

        start = time.time()
        HORt2 = 2*np.pi*mb*lam2p*1/s2*mcint__H1 - mb*lam2p/s2*mcint__H3
        end = time.time()
        etime2 = end - start
        
        return [s1, s2, HORt1, HORt2]    ## return a line of row
    else: 
        return [s1, s2, None, None]    ## The case either s1 or s2 is zero is not included in this calculus.

    #### out_HOR ends: the function for calculating HOR (the handover rate) ####


    
#### main function ####

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
        print('                2nd: PCP indicator')
        print('                     (TCP(tcp) <- Thomas Cluster Process / (MCP) <- Matern Cluster Process)')
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

    PCP_id = argvs[2].upper()
    if PCP_id not in ['TCP', 'MCP']:
        print('InputError: Please set a valid PCP_id; TCP(tcp)/MCP(mcp).')
        print('            (TCP(tcp) <- Thomas Cluster Process / (MCP) <- Matern Cluster Process)')
        sys.exit()


    result_header = ['skipping time1 (macro)', 'skipping time2 (small)', 'handover rate1 (macro)', 'handover rate2 (small)']
    result_rows_L = []
    NoticeMessagesFlag = False
    for i in range(len(matrix)):
        ParamSets, CurveType, ErrorFlag = make_parameters_set(matrix[i])

        if not ErrorFlag:
            result_row = output_HOR(ParamSets, PCP_id)
        else:
            NoticeMessagesFlag = True
            result_row = ['', '']
            continue
        result_rows_L.append(result_row)
    if NoticeMessagesFlag:
        NoticeMessages()


    OutputCsvFilename = 'result_calc-HOR-approx-{}_'.format(PCP_id) + InputCsvFilename
    f = open(OutputCsvFilename, 'w')
    csvwriter = csv.writer(f)
    csvwriter.writerow(result_header)
    csvwriter.writerows(result_rows_L)
    f.close()
