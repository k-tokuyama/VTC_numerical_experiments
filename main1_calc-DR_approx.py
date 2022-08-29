
import sys
import csv
import numpy as np
import time

from scipy import integrate
from scipy import stats



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


#def output_DR(param_sets, PCP_id, flag_sobol, McNum_sobol=1024, DivNum_HOR=100, DivNum_DR1=400, DivNum_DR2=20, DivNum_DR3=4):
def output_DR(param_sets, PCP_id):

    ### Required parameters.
    if PCP_id == 'TCP':
        s1, s2, veloc, lam1, lam2p, mb, sigma, P1, P2, beta = param_sets
    elif PCP_id == 'MCP':
        s1, s2, veloc, lam1, lam2p, mb, rd, P1, P2, beta = param_sets
    else:
        print('Error, output_DR: Please input a valid PCP_id; "TCP" or "MCP". ')
        return None

    if not s1:
        s1 = 10**(-8)
    if not s2:
        s2 = 10**(-8)

    ### Division numbers for trapezoidal integrals.
    DivNum_theta = 4
    DivNum_y = 6
    DivNum_r = 100
    DivNum_z = 100
    DivNum_x_type1, DivNum_x_type2 = 100, 20


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
                L_x = np.linspace(min(r, abs(rd - z)), min(r, rd + z), DivNum_x_type1+1)
                return (min(r, max(rd - z, 0))**2 + 2/np.pi*integrate.trapz(integd_x(L_x, z), L_x))/rd**2



    ### Functions that appear in the Data Rate expression.

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
    def wcos(r, l, theta):
        return np.sqrt(r**2 + l**2 - 2*r*l*np.cos(theta))
    def R_til(x, h, l):
        if not x:
            x = 10**(-8)
        if not l:
            l = 10**(-8)
        return np.arccos(np.clip((x**2+l**2-h**2)/(2*x*l),-1,1))
    

    

    def K_beta(beta):
        return np.pi**2/beta/np.sin(2*np.pi/beta)

    def func_rho(tier_i, y, r, u, theta):
        def integd_x(L_x):
            return np.array([ x*R_til(x, P_bar(tier_i, 1)*r, u) * ( 1 + x**beta/(P_bar(tier_i, 1)**beta*wcos(r,u,theta)**beta*y) )**(-1) for x in L_x])
        L_x = np.linspace(0, u + P_bar(tier_i, 1)*r, DivNum_x_type1+1)
        epnt_part1 =  K_beta(beta)*P_bar(tier_i, 1)**2*wcos(r,u,theta)**2*y**(2/beta) 
        if not y or not r or not u:
            epnt_part2 = 0.0
        else:
            epnt_part2 = integrate.trapz(integd_x(L_x), L_x)
        return np.exp( -2*lam1*(epnt_part1 - epnt_part2) )

    def func_K_mod(tier_i, y, r, u, theta, z):
        def integd_x_cv(L_x_cv):
            def integd_x(x):
                frcinv = ( 1 + x**beta/(P_bar(tier_i, 2)**beta * wcos(r,u,theta)**beta * y) )**(-1)
                return func_fd(x, z) * frcinv
            L_x = np.tan(L_x_cv)
            return np.array([ (1 + x**2)*integd_x(x) for x in L_x ])

        ## determine the lower/upper bound of the integral range w.r.t. x (by truncation).
        if PCP_id == 'MCP':
            x_LowerBound, x_UpperBound = max(z - rd, 0.0), z + rd
        elif PCP_id == 'TCP':
            rd_trunc = stats.rayleigh.ppf(q=0.99, loc=0.0, scale=sigma)
            x_LowerBound, x_UpperBound = max(z - rd_trunc, 0.0), z + rd_trunc
        else:
            print('Error, func_K_mod: Please input a valid PCP_id; "TCP" or "MCP". ')
            return None

        L_x_cv_truncated = np.linspace(np.arctan(x_LowerBound), np.arctan(x_UpperBound), DivNum_x_type2+1)
        return np.exp( -mb * integrate.trapz(integd_x_cv(L_x_cv_truncated), L_x_cv_truncated) )


    def func_A(tier_i, r):
        def integd_z_cv(L_z_cv, r):
            def integd_z(z):
                return z*(1 - np.exp( -mb*func_Fd(P_bar(tier_i, 2)*r, z) ))
            L_z = np.tan(L_z_cv)
            return np.array([ (1 + z**2)*integd_z(z) for z in L_z ])

        ## determine the lower/upper bound of the integral range w.r.t. z (by truncation).
        if PCP_id == 'MCP':
            z_LowerBound, z_UpperBound = 0.0, r + rd
        elif PCP_id == 'TCP':
            rd_trunc = stats.rayleigh.ppf(q=0.99, loc=0.0, scale=sigma)
            z_LowerBound, z_UpperBound = 0.0, r + rd_trunc
        else:
            print('Error, func_K_mod: Please input a valid PCP_id; "TCP" or "MCP". ')
            return None

        L_z_cv_truncated = np.linspace(np.arctan(z_LowerBound), np.arctan(z_UpperBound), DivNum_z+1)
        return np.exp( -2*np.pi*lam2p * integrate.trapz(integd_z_cv(L_z_cv_truncated, r), L_z_cv_truncated) )

    def func_B(r):
        def integd_z_cv(L_z_cv, r):
            def integd_z(z):
                return z*func_fd(r, z)*np.exp( -mb*func_Fd(r, z) )
            L_z = np.tan(np.pi/2*L_z_cv)
            return np.array([ np.pi/2*(1 + z**2)*integd_z(z) for z in L_z ])
        L_z_cv = np.linspace(10**(-8), 1, DivNum_z+1)
        return 2*np.pi*lam2p * integrate.trapz(integd_z_cv(L_z_cv, r), L_z_cv)

    def func_J(tier_i, y, r, u, theta):
        def integd_z_cv(L_z_cv, r):
            #def integd_z(z):
            #    return z*(2 - np.exp(-mb*func_Fd(P_bar(tier_i, 2)*r, z)) - func_K_mod(tier_i, y, r, u, theta, z))
            def integd_z(z):
                return z*(1 - func_K_mod(tier_i, y, r, u, theta, z))
            L_z = np.tan(np.pi/2*L_z_cv)
            return np.array([ np.pi/2*(1 + z**2)*integd_z(z) for z in L_z ])
        L_z_cv = np.linspace(0, 1, DivNum_z+1)
        return func_A(tier_i, r) * np.exp( -2*np.pi*lam2p * integrate.trapz(integd_z_cv(L_z_cv, r), L_z_cv) )



    ### Computation of the Data Rate expression

    def summand_DR_term1(t1):
        def integd_y_cv(L_y_cv):
            def integd_y(y):
                def integd_r_cv(L_r_cv, y):
                    def integd_r(r, y):
                        def integd_theta(L_theta, r, y):
                            return np.array([ func_rho(1, y, r, t1*veloc, theta)*func_J(1, y, r, t1*veloc, theta) for theta in L_theta ])
                        L_theta = np.linspace(0, np.pi, DivNum_theta+1)
                        return r*np.exp(-lam1*np.pi*r**2)*integrate.trapz(integd_theta(L_theta, r, y), L_theta)
                    L_r = np.tan(np.pi/2*L_r_cv)
                    return np.array([ np.pi/2*(1 + r**2)*integd_r(r, y) for r in L_r ])
                L_r_cv = np.linspace(10**(-8), 1, DivNum_r+1)
                return integrate.trapz(integd_r_cv(L_r_cv, y), L_r_cv)/(1 + y)
            L_y = np.exp(np.sinh(L_y_cv)*np.pi/2)
            return np.array([ (np.cosh(y_cv)*np.pi/2)*y*integd_y(y) for y_cv, y in zip(L_y_cv, L_y) ])
        y_cv_min = -3.0; y_cv_max = 3.0
        L_y_cv = np.linspace(y_cv_min, y_cv_max, DivNum_y+1)
        return 2*lam1*integrate.trapz(integd_y_cv(L_y_cv), L_y_cv)


    def summand_DR_term2(t2):
        def integd_y_cv(L_y_cv):
            def integd_y(y):
                def integd_r_cv(L_r_cv, y):
                    def integd_r(r, y):
                        def integd_theta(L_theta, r, y):
                            return np.array([ func_rho(2, y, r, t2*veloc, theta)*func_J(2, y, r, t2*veloc, theta) for theta in L_theta ])
                        L_theta = np.linspace(0, np.pi, DivNum_theta+1)
                        return np.exp(-lam1*np.pi*P_bar(2, 1)**2*r**2)*func_B(r)*integrate.trapz(integd_theta(L_theta, r, y), L_theta)
                    L_r = np.tan(L_r_cv)
                    return np.array([ (1 + r**2)*integd_r(r, y) for r in L_r ])

                ## determine the upper bound of the integral range w.r.t. r (by truncation).
                r_UpperBound_term2 = 0.3 + 10**(-8)

                L_r_cv_truncated = np.linspace(10**(-8), np.arctan(r_UpperBound_term2), DivNum_r+1)
                return integrate.trapz(integd_r_cv(L_r_cv_truncated, y), L_r_cv_truncated)/(1 + y)
            L_y = np.exp(np.sinh(L_y_cv)*np.pi/2)
            return np.array([ (np.cosh(y_cv)*np.pi/2)*y*integd_y(y) for y_cv, y in zip(L_y_cv, L_y) ])
        y_cv_min = -3.0; y_cv_max = 3.0
        L_y_cv = np.linspace(y_cv_min, y_cv_max, DivNum_y+1)
        return mb/np.pi*integrate.trapz(integd_y_cv(L_y_cv), L_y_cv)


    start1 = time.time()
    DR_term1 = summand_DR_term1(s1)
    end1 = time.time()
    e_time1 = end1 - start1

    start2 = time.time()
    DR_term2 = summand_DR_term2(s2)
    end2 = time.time()
    e_time2 = end2 - start2

    return [s1, s2, DR_term1, DR_term2], [e_time1, e_time2]



    
#### main function ####

def NoticeMessages():
    print('<< Notice >>')
    print('11 arguments are required to be written in the input csv.')
    print('the 11 args -> 1 : skipping time for ppp')
    print('               2 : skipping time for pcp')
    print('               3 : moving velocity of a user')
    print('               4 : intensity for ppp')
    print('               5 : parent intensity for pcp')
    print('               6 : daughter intensity for pcp')
    print('               7 : daighter variance for pcp')
    print('               8 : transmitting power for ppp')
    print('               9 : transmitting power for pcp')
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


    ## Make the result csv from the input csv.
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


    ## The matrix data of the result csv.
    result_rows_L = []
    NoticeMessagesFlag = False
    for i in range(len(matrix)):
        ParamSets, CurveType, ErrorFlag = make_parameters_set(matrix[i])
        if not ErrorFlag:
            result_row1, result_row3 = output_DR(ParamSets, PCP_id)
        else:
            NoticeMessagesFlag = True
            result_row = ['', '']
            continue
        if not i:
            result_row2 = [None, None]
        else:
            result_row2 = np.mean(np.array(result_rows_L)[:, 2:4], axis=0).tolist()
        result_rows_L.append(result_row1 + result_row2)
    if NoticeMessagesFlag:
        NoticeMessages()

    ## Header data of the result csv.
    header1 = ['skipping time1 (macro)', 'skipping time2 (small)', 'instantaneous data rate1 (macro)', 'instantaneous data rate2 (small)']
    header2 = ['average data rate1 (macro)', 'average data rate2 (small)']
    header3 = ['elapsed time1 (macro)', 'elapsed time2 (small)']
    result_header = header1 + header2


    ## Output the result csv file.
    OutputCsvFilename = 'result_calc-DR-approx-{}_'.format(PCP_id) + InputCsvFilename
    f = open(OutputCsvFilename, 'w')
    csvwriter = csv.writer(f)
    csvwriter.writerow(result_header)
    csvwriter.writerows(result_rows_L)
    f.close()




