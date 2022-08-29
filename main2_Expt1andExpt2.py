
## argvs[1]: PCP_id; TCP(tcp)/MCP(mcp)
## argvs[2]: csv_filename_id
##  - The following two csv files must exist in this directory.
##       - 1: result_calc-DR-approx-<PCP_id>_<csv_filename_id>.csv
##       - 2: result_calc-HOR-approx-<PCP_id>_<csv_filename_id>.csv

import time
import sys
import csv
import numpy as np



def read_csv(CsvFileName):
    f = open(CsvFileName, 'r')
    csvreader = csv.reader(f)
    header = next(csvreader)
    ## The first row of the matrix is omitted.
    OmittedRow = next(csvreader)
    matrix = [list(map(float, v)) for v in csvreader]
    return header, matrix

def write_csv(CsvFileName, CsvHeader, CsvMatrix):
    f = open(CsvFileName, 'w')
    csvwriter = csv.writer(f)
    csvwriter.writerow(CsvHeader)
    csvwriter.writerows(CsvMatrix)
    f.close()


def NoticeMessage():
    print('---Notice---')
    print('The following arguments are required for the inputs:')
    print('    argvs[1]: PCP_id; TCP(tcp)/MCP(mcp)')
    print('    argvs[2]: csv_filename_id; result_calc-DR-approx-<PCP_id>_<csv_filename_id>.csv')

if __name__ == '__main__':
    
    argvs = sys.argv

    if not len(argvs[1:]) == 2:
        print('InputError: Please input the 2(two) arguments in the input command.')
        NoticeMessage()
        sys.exit()
    else:

        if argvs[1].upper() in ['TCP', 'MCP']:
            PCP_id = argvs[1].upper()
        else:
            print('Error, argvs[1]: Please input the valid PCP_id for the 1st argument; TCP(tcp)/MCP(mcp).')
            NoticeMessage()
            sys.exit()

        csv_filename_id = argvs[2].split('.')[0]
        try:
            Filename1 = 'result_calc-DR-approx-{}_{}.csv'.format(PCP_id, csv_filename_id)
            f = open(Filename1, 'r')
            f.close()
        except FileNotFoundError:
            print('Error-DR-input, argvs[1],[2]: The corresponding result DR-approx csv is not found in this directory.')
            print('                              Please input the valid PCP_id, csv_filename_id;')
            print('                                   - result_calc-DR-approx-<PCP_id>_<csv_filename_id>.csv')
            NoticeMessage()
            sys.exit()
        try:
            Filename2 = 'result_calc-HOR-approx-{}_{}.csv'.format(PCP_id, csv_filename_id)
            f = open(Filename2, 'r')
            f.close()
        except FileNotFoundError:
            print('Error-HOR-input, argvs[1],[2]: The corresponding result HOR-exact csv is not found.')
            print('                               Please input the valid PCP_id, csv_filename_id;')
            print('                                    - result_calc-HOR-approx-<PCP_id>_<csv_filename_id>.csv')
            NoticeMessage()
            sys.exit()
        

    j_DR_macro, j_DR_small = 4, 5
    j_HOR_macro, j_HOR_small = 2, 3

    header1, matrix1 = read_csv(Filename1)
    header2, matrix2 = read_csv(Filename2)

    result_header = ['s2', 'handover rate', 'data rate']
    result_matrix = []

    if len(matrix1) == len(matrix2):
        i_macro = len(matrix1) - 1    ## i_macro refers to the lowest row of the matrix1. 
        if matrix1[i_macro][0] == matrix2[i_macro][0]:
            s_macro = matrix1[i_macro][0]

            for i_small in range(len(matrix1)):
                if matrix1[i_small][1] == matrix2[i_small][1]:
                    s_small = matrix1[i_small][1]

                    DR_macro, DR_small = matrix1[i_macro][j_DR_macro], matrix1[i_small][j_DR_small]
                    HOR_macro, HOR_small = matrix2[i_macro][j_HOR_macro], matrix2[i_small][j_HOR_small]

                    result_matrix.append([s_small, HOR_macro + HOR_small, DR_macro + DR_small])

                else:
                    print("Error, check3: the values of s_small in the two csv's argvs[1] and argvs[2] are not consistent.")
                    sys.exit()

        else:
            print("Error, check2: the values of s_macro in the two csv's argvs[1] and argvs[2] are not consistent.")
            sys.exit()
    else:
        print("Error, check1: the number of rows of argvs[1] and argvs[2] csv's are not consistent.")
        sys.exit()

    write_csv('result_Expt1andExpt2_{}_{}.csv'.format(PCP_id, csv_filename_id), result_header, result_matrix)


