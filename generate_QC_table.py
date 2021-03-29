# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 14:21:28 2021

@author: rjovelin
"""


import os
import json
import argparse



def collect_metrics(smmipdir):
    '''
    (str) -> dict
    
    Return a dictionary with QC metrics extracted from the "extraction_metrics.json" file for each sample
    
    Parameters
    ----------
    - smmipdir (str): Directory with the sample sub-directories containing the QC metric files
    '''
        
    #smmipdir = '/scratch2/groups/gsi/bis/rjovelin/smmips/{0}/smmips_analysis'.format(run)

    samples = [i for i in os.listdir(smmipdir)]

    # make a table summarizing QC 
    metrics = [(os.path.join(smmipdir, '{0}/stats/{0}_extraction_metrics.json'.format(i)), i) for i in os.listdir(smmipdir)] 
    # remove non-exisiting files
    print(len(metrics))
    to_remove = [i for i in metrics if os.path.isfile(i[0]) == False]
    for i in to_remove:
        metrics.remove(i)
    print(len(metrics))

    # create a dictionary
    D = {}
    for i in metrics:
        sample = i[1]
        infile = open(i[0])
        d = json.load(infile)
        infile.close()
        assert sample not in D
        D[sample] = d
    return D

def write_summary(outputfile, d):
    newfile = open(outputfile, 'w')
    header = ["library", "reads", "assigned", "percent_assigned", "not_assigned", "percent_not_assigned", "assigned_empty", "percent_empty_smmips"]
    newfile.write('\t'.join(header) + '\n')          
    for sample in d:
        line = [sample]
        for i in header[1:]:
            line.append(d[sample][i])
        line = list(map(lambda x: str(x), line))
        newfile.write('\t'.join(line) + '\n')
    newfile.close()


def generate_qc_table(smmipdir, outputfile):
    '''
    (str, str) -> None
    
    Write a QC summary table with metrics for all samples located in smmipdir
    
    Parameters
    ----------
    - smmipdir (str): Directory with sample sub-directories containing the out, stats, alignments directories
    - outputfile (str): Name of the output table
    '''
       
    # collect QC data for each sample in a given run
    D = collect_metrics(smmipdir)
    # write summary file
    write_summary(outputfile, D)




if __name__ == '__main__':

    # create top-level parser
    parser = argparse.ArgumentParser(prog = 'smmipQCTable.py', description='Generates a summary table with sample QC metrics', add_help=True)
    
    parser.add_argument('--table', dest='outputfile', help='Name of the output QC summary table', required=True)
    parser.add_argument('--smmipdir', dest='smmipdir', help='Run ID', required=True)
        
    # get arguments from the command line
    args = parser.parse_args()
    
    generate_qc_table(args.smmipdir, args.outputfile)        
    