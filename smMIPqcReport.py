# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 08:40:03 2021

@author: rjovelin
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import matplotlib as mpl
from matplotlib.lines import Line2D
import argparse
import markdown
from datetime import datetime
import xhtml2pdf.pisa as pisa
import os


def read_qc_table(qc_table):
    '''
    (str) -> dict
    
    Return a dictionary with all QC metrics per sample
    
    Parameters
    ----------
    - qc_table (str): Path to summary file with sample QC metrics
    '''
    
    infile = open(qc_table)
    header = infile.readline().rstrip().split('\t')
    D = {}
    for line in infile:
        line = line.rstrip()
        if line != '':
            line = line.split('\t')
        sample = line[0]
        d = {}
        for i in range(1, len(header)):
            d[header[i]] = line[i]
        for i in d:
            if '.' in d[i]:
                d[i] = float(d[i])
            else:
                d[i] = int(d[i])
        D[sample] = d
    infile.close()
    return D


def get_samples_qc(D):
    '''
    (dict) -> (list, list, list, list, list, list, list)
    
    Returns a list of samples metrics, sorted by percent 
    assigned reads and for which ith element is the QC metric of the ith element
    of the sample list, the first list in the tuple.
    
    Parameters
    ----------
    - D (dict): Dictionary with QC metrics for each sample
    '''
    
    # make a list of samples sorted according to read counts
    # all data will be plotted according to increasing values of read counts
    L = sorted([(D[sample]['percent_assigned'], sample) for sample in D])
    L.reverse()
    # make a list of read counts
    percent_assigned = [i[0] for i in L]
    # make a list of samples
    samples = [i[1] for i in L]

    # make a list of read_counts
    read_counts = [D[i]['reads'] for i in samples]
    # make a list of percent discarded
    percent_discarded = [D[i]['percent_not_assigned'] for i in samples]
    # make a list of assigned reads
    assigned = [D[i]['assigned'] for i in samples]
    # make a list of empty smmips
    empty = [D[i]['assigned_empty'] for i in samples]
    # make a list of percent empty smmips
    percent_empty = [D[i]['percent_empty_smmips'] for i in samples]

    return samples, percent_assigned, read_counts, assigned, percent_discarded, empty, percent_empty


def CreateAx(row, col, pos, figure, Data, samples, run, YLabel, title = None, XLabel = None, line = None):
    
    ax = figure.add_subplot(row, col, pos)
    
    
    #xcoord = [i/10 for i in range(len(Data))]
    xcoord = []
    x = 0
    for i in range(len(Data)):
        xcoord.append(x)
        x += 0.2
    ax.bar(xcoord, Data, width=0.2, color='grey', edgecolor='black', linewidth=2, alpha=1)
             
    # add line Y = 75% or line = 25%
    if line == 'high':
        ax.axhline(y=75, color='r', linestyle='-', linewidth=3, alpha=0.5)
#    elif line == 'low':
#        ax.axhline(y=25, color='r', linestyle='-', linewidth=3, alpha=0.5)
        
    # write axis labels
    if XLabel is not None:
        ax.set_xlabel(XLabel, color='black', size=38, ha='center', weight= 'normal')
    ax.set_ylabel(YLabel, color='black', size=38, ha='center', weight='normal')

    # add title 
    if title is not None:
        ax.set_title(title, weight='bold', pad =20, fontdict={'fontsize':40})

    # add ticks labels
    if XLabel is not None:
        
        print(samples)
        
        plt.xticks(xcoord, [i[:i.index('_TS')+len('_TS')] if '_TS' in i else i for i in samples], ha='center', fontsize=12, rotation='vertical')
    else:
        plt.xticks(xcoord, ['' for i in samples], ha='center', fontsize=12, rotation=0)

    # add splace bewteen axis and tick labels
    ax.yaxis.labelpad = 17
    ax.xaxis.labelpad = 17
    
    # do not show frame lines  
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(True)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)    
        
    # offset the x axis
    for loc, spine in ax.spines.items():
        spine.set_position(('outward', 5))
        spine.set_smart_bounds(True)
    
    # keep ticks only along the x axis, edit font size and change tick direction
    if XLabel is not None:
        plt.tick_params(axis='both', which='both', bottom=True, top=False, right=False, left=False,
                    labelbottom=True, colors = 'black', labelsize = 12, direction = 'out')
        plt.tick_params(axis='x', which='both', bottom=True, top=False, right=False, left=False,
                    labelbottom=True, colors = 'black', labelsize = 12, direction = 'out', labelrotation=90)
        plt.tick_params(axis='y', which='both', bottom=True, top=False, right=False, left=False,
                    labelbottom=True, colors = 'black', labelsize = 20, direction = 'out')
    else:
        plt.tick_params(axis='both', which='both', bottom=True, top=False, right=False, left=False,
                    labelbottom=False, colors = 'black', labelsize = 20, direction = 'out')
        
    # add a light grey horizontal grid to the plot, semi-transparent, 
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.7, linewidth = 0.5)  
    # hide these grids behind plot objects
    ax.set_axisbelow(True)

    return ax



def plot_qc_metrics(project, run, plate, samples, percent_assigned, read_counts, assigned, percent_discarded, empty, percent_empty, Title):
    
    
    figure = plt.figure(1, figsize = (55, 55)) 

    # bar grap percent assigned
    ax1 = CreateAx(6, 1, 1, figure, percent_assigned, samples, run, 'percent assigned', title = Title, XLabel = None, line = 'high')
    # bar graphs with read counts
    ax2 = CreateAx(6, 1, 2, figure, read_counts, samples, run, 'reads', title = None, XLabel = None, line = None)
    # bar grap pre-assigned
    ax3 = CreateAx(6, 1, 3, figure, assigned, samples, run, 'assigned', title = None, XLabel = None, line = None)
    # bar graph with percent discarded
    ax4 = CreateAx(6, 1, 4, figure, percent_discarded, samples, run, 'percent unassigned', title = None, XLabel = None, line = None)
    # bar empty smmips
    ax5 = CreateAx(6, 1, 5, figure, empty, samples, run, 'assigned empty', title = None, XLabel = None, line = None)
    # bar percent empty smmips
    ax6 = CreateAx(6, 1, 6, figure, percent_empty, samples, run, 'percent empty', title = None, XLabel = 'libraries', line = None)

    plt.tight_layout()  

    # save figure file
    if project is None:
        project = ''
    if plate is None:
        plate = ''
    outputfile = '_'.join([project, run, plate.replace(' ', '_'), 'smMIP_QC.png']).strip('_')
    figure.savefig(outputfile, bbox_inches = 'tight')
    plt.close()
    return outputfile

def get_sample_plate_location(sample_wells):
    '''
    (str) -> dict
    
    Returns a dictionary with plate and well location for each sample
    
    Parameters
    ----------
    - sample_wells (str): Path to file with plate and well information for each sample
    '''
    
    # get the plates
    infile = open(sample_wells)
    samples_plates = {}
    for line in infile:
        line = line.rstrip()
        if line != '':
            line = line.split('\t')
            sample = line[0].strip()
            if 'libraries' in line[1]:
                plate, well = line[1].split('libraries')
            elif 'Libraries' in line[1]:
                plate, well = line[1].split('Libraries')        
            elif 'Library box' in line[1]:
                plate, well = line[1].split('Library box')
            elif 'Control Library' in line[1]:
                plate, well = line[1].split('Control Library')
            plate = plate.strip()
            well = well.replace('-', '').strip()
            assert sample not in samples_plates
            samples_plates[sample] = [plate, well]
    infile.close()
    return samples_plates


def get_plates(samples_plates, D, project):
    '''
    (dict, dict, str) -> list
    
    Return a list of plates with samples of interest
    
    Parameters
    ----------
    - samples_plates (dict): Dictionary with plate and well location 
    - D (str): Dictionary with QC metrics for each sample
    - project (str): Name of the project. Expected in the sample name.
                     None if all samples from all projects are processed together
    '''
    
    # make a list of plates
    plates = sorted(list(set([samples_plates[i][0] for i in samples_plates if i in D])))
    # create a dict to store samples on each plate
    s = {}
    for i in samples_plates:
        if samples_plates[i][0] not in s:
            s[samples_plates[i][0]] = []
        s[samples_plates[i][0]].append(i)
    # check if plates only contains control samples
    if project:
        to_remove = []
        for i in s:
            contains_samples = []
            for j in s[i]:
                contains_samples.append(project in j)
            # remove plate if only controls are on the plate
            if not any(contains_samples):
                to_remove.append(i)
        for i in to_remove:
            if i in plates:
                plates.remove(i)
    return plates


def map_samples_to_wells(samples_plates, current_plate):
    '''
    (dict, str) -> dict
    
    Returns a dictionary with sample ID for each well on current_plate
    
    Parameters
    ---------
    - samples_plates (dict): Dictionary with plate, well info for each sample
    - current_late (str): Plate of interest
    '''
    
    # get well information
    wells = {}
    for i in samples_plates:
        if samples_plates[i][0] == current_plate:
            wells[samples_plates[i][-1]] = i
    
    all_wells = ['A01',  'A02',  'A03',  'A04',  'A05',  'A06',  'A07',  'A08', 'A09',  'A10',  'A11',  'A12',
                 'B01',  'B02',  'B03',  'B04',  'B05',  'B06',  'B07',  'B08',  'B09',  'B10',  'B11',  'B12', 
                 'C01',  'C02',  'C03',  'C04',  'C05',  'C06',  'C07',  'C08',  'C09',  'C10',  'C11',  'C12',
                 'D01',  'D02',  'D03',  'D04',  'D05',  'D06',  'D07',  'D08',  'D09',  'D10',  'D11',  'D12',
                 'E01',  'E02',  'E03',  'E04',  'E05',  'E06',  'E07',  'E08',  'E09',  'E10',  'E11',  'E12',
                 'F01',  'F02',  'F03',  'F04',  'F05',  'F06',  'F07',  'F08',  'F09',  'F10',  'F11',  'F12',
                 'G01',  'G02',  'G03',  'G04',  'G05',  'G06',  'G07',  'G08',  'G09',  'G10',  'G11',  'G12',
                 'H01',  'H02',  'H03',  'H04',  'H05',  'H06',  'H07',  'H08',  'H09',  'H10',  'H11',  'H12']

    for i in all_wells:
        if i not in wells:
            wells[i] = ''

    return wells


def samples_on_plate(wells):
    '''
    (dict) -> list
    
    Returns a 2D list with sample names in each well
    
    Parameters
    ----------
    - wells (dict): Dictionary with sample name for each well
    '''
    
    # make a list of samples sorted by positions [[A01, A02, A12], [B01, ..., B12]]
    keys = []
    for i in 'ABCDEFGH':
        k = []
        for j in range(1, 13):
            if j < 10:
                k.append(i + '0' + str(j))
            else:
                k.append(i + str(j))
        keys.append(k)
    # make a list of samples corresponding to each well
    samples = []
    for i in keys:
        k = []
        for j in i:
            k.append(wells[j])
        samples.append(k)
    return samples



def qc_metrics_on_plate(D, samples, metrics):
    '''
    (dict, list, str) -> numpy.ndarray
    
    Returns a numpy array with QC metrics of interest in each well
    
    Parameters
    ----------
    - D (dict): Dictionary with QC metrics for each sample
    - samples (list): 2D list with sample name in each well
    - metrics (str): Metrics of interest. Values:
                     reads, assigned, percent_assigned, percent_empty_smmips
    '''
    
    # make a 2D list of qc metrics [[A01, A02, A12], [B01, ..., B12]]
    L = []
    for i in samples:
        k = []
        for j in i:
            if j in D:
                k.append(D[j][metrics])
            else:
                k.append(0)
        L.append(k)
    L = np.array(L)    
    return L
    

def create_ax_heatmap(row, col, pos, figure, Data, XLabel, title=None):

    ax = figure.add_subplot(row, col, pos)

    L = []
    for i in Data:
        for j in i:
            L.append(j)
    # plot heatmap (use vmin and vmax to get the full range of values)
    heatmap = ax.imshow(Data, interpolation = 'nearest', cmap = 'YlGn', vmin=min(L), vmax=max(L))
        
    # add heatmap scale 
    cbar = plt.colorbar(heatmap, shrink=0.60)
    # edit tcik parameters of the heatmap scale
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.tick_params(direction = 'out')
    # edit xticks
    xtick_labels = []
    for j in range(1, 13):
        if j < 10:
            xtick_labels.append('0' + str(j))
        else:
            xtick_labels.append(str(j))
    plt.xticks([i for i in range(12)], xtick_labels)
    plt.yticks([i for i in range(9)], ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
    # do not show lines around figure  
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)  
    
    ax.set_xlabel(XLabel, color='black', size=16, ha='center', weight= 'normal')
    ax.xaxis.labelpad = 15
    
    if title is not None:
        ax.set_title(title, weight='bold', pad =20, fontdict={'fontsize':20})
    
    # edit tick parameters    
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
                    right = False, left = False, labelbottom=True, labelleft = True,
                    colors = 'black', labelsize = 14, direction = 'out')  
    
    return ax



def plot_heatmaps(read_counts, assigned, percent_assigned, percent_empty, project, run, current_plate):
    
    figure = plt.figure(1, figsize = (25, 20)) 
    # plot data
    ax1 = create_ax_heatmap(4, 1, 1, figure, read_counts, 'read count', title=current_plate)
    ax2 = create_ax_heatmap(4, 1, 2, figure, assigned, 'assigned reads', title=None)
    ax3 = create_ax_heatmap(4, 1, 3, figure, percent_assigned, 'percent assigned', title=None)
    ax4 = create_ax_heatmap(4, 1, 4, figure, percent_empty, 'percent empty', title=None)

    # make sure subplots do not overlap
    plt.tight_layout()
    # save figure
    plate_name = current_plate.strip().replace(' ', '_')
    if project is None:
        project = ''
    figure_name = '_'.join([project, run, plate_name, 'QC_heatmap.png']).strip('_')
    figure.savefig(figure_name, bbox_inches = 'tight')
    plt.close()
    return figure_name


def keep_project_samples(D, project):
    '''
    (dict, str) -> dict
    
    Return a dictionary without samples not included in project unless samples are GLCS controls
    '''
    
    to_remove = [i for i in D if project not in i and 'GLCS' not in i]
    for i in to_remove:
        del D[i]
    return D




def generate_report(project, run, QC_summary, sample_wells, include_plate):
    '''
    (str | None, str, str, str, bool)
    

    Parameters
    ----------    
        
    - project (str | None): Name of the project
    - run (str): Run ID
    - QC_summary (str): Path to the summary file with QC metrics
    - sample_wells (str): Path to the file with sample plate and well information
    - include_plate (bool): Include metrics plots at the plate level and heatmaps if True
    '''
    
    # extract QC metrics
    D = read_qc_table(QC_summary)
    # remove samples not in project, but keep controls
    if project:
        D = keep_project_samples(D, project)
    
    # get the qc table header
    infile = open(QC_summary)
    header = infile.readline().rstrip().split('\t')
    infile.close()
    
    # get the samples of interest and associated qc metrics
    samples, percent_assigned, read_counts, assigned, percent_discarded, empty, percent_empty = get_samples_qc(D)

    # plot qc metrics for the run and project
    metrics_figure = plot_qc_metrics(project, run, '', samples, percent_assigned, read_counts, assigned, percent_discarded, empty, percent_empty, '{0} smMip QC'.format(run))
    
    # get the samples plate and well location
    samples_plates = get_sample_plate_location(sample_wells)
    
    # check that all samples have well info
    missing = []
    for i in D:
        if i not in samples_plates:
            print(i)
            missing.append(i)
    assert len(missing) == 0
    
    
    #assert sorted([i for i in D]) == sorted([i for i in samples_plates])
    
    # make a list of plates
    plates = get_plates(samples_plates, D, project)

    # collect the name of the fugure files with heatmaps
    heatmap_names = []
    metric_plots = []
    
    for current_plate in plates:
        # keep only samples on that plate
        Dprime = {i:D[i] for i in D if samples_plates[i][0] == current_plate}
        # get the samples of interest and associated qc metrics
        s, pa, rc, a, pd, e, pe = get_samples_qc(Dprime)
        metrics_figure_plate = plot_qc_metrics(project, run, current_plate, s, pa, rc, a, pd, e, pe, current_plate)
        metric_plots.append(metrics_figure_plate)
        # map samples with qc metrics to each well
        wells = map_samples_to_wells(samples_plates, current_plate)
        # add 0 values to wells without samples
        Dprime[''] = {header[i]:0 for i in range(1, len(header))}
        # make a 2D list of samples corresponding to each well
        current_samples = samples_on_plate(wells)
        # get 2D array with qc metrics matching samples in each well
        read_counts = qc_metrics_on_plate(Dprime, current_samples, 'reads')
        assigned = qc_metrics_on_plate(Dprime, current_samples, 'assigned')
        percent_assigned = qc_metrics_on_plate(Dprime, current_samples, 'percent_assigned')
        percent_empty = qc_metrics_on_plate(Dprime, current_samples, 'percent_empty_smmips')
        # plot heatmaps for the current plate
        figure_name = plot_heatmaps(read_counts, assigned, percent_assigned, percent_empty, project, run, current_plate)
        heatmap_names.append(figure_name)


    # get current date (year-month-day)
    current_date = datetime.today().strftime('%Y-%m-%d')
    
    # make a list to store the text of the report
    L = []
    
    # add title
    L.append('# smMIP QC Report')
    
    if project:
        L.append('Project: {0}'.format(project))          
             
    L.append('\n')
    L.append('## Run: {0}'.format(run))         
    L.append('Date: {0}'.format(current_date) + '\n')                
           
    L.append('Sample QC metrics are available in the table: {0}'.format(os.path.basename(QC_summary)))

                    
    L.append('### Plots of QC metrics per sample\n')

    L.append('![QC_metrics]({0})'.format(metrics_figure))
    
    L.append('\n')
    
    if include_plate:
        L.append('Plots of QC metrics per plate\n')
        for i in metric_plots:
            L.append('![qc]({0})'.format(i))
            L.append('\n')
        L.append('Plots of QC metrics per plate and well location\n')
        for i in heatmap_names:
            L.append('![heatmap]({0})'.format(i))
            L.append('\n')
    # convert list to string text
    content = '\n'.join(L)
    
    # convert markdown to html
    html = markdown.markdown(content)
     
    if project is None:
        report_file = 'smMIP_QC_Report_{0}.pdf'.format(run)
    else:
        report_file = 'smMIP_QC_Report_{0}_{1}.pdf'.format(project, run)
    # convert html to pdf
    newfile = open(report_file, "wb")
    pisa.CreatePDF(html, newfile)
    newfile.close()


if __name__ == '__main__':

    # create top-level parser
    parser = argparse.ArgumentParser(prog = 'smMIPqcReport.py', description='Generates a report for the smMIP QC analysis', add_help=True)

    
    parser.add_argument('-p', '--Project', dest='project', help='Name of the project')
    parser.add_argument('-r', '--Run', dest='run', help='Run ID', required=True)
    parser.add_argument('-q', '--QC', dest='qc', help='Path to the table with sample QC metrics', required=True)
    parser.add_argument('-l', '--Location', dest='location', help='Path to the file with sample plate and well information')
    parser.add_argument('--plates', dest='include_plates', action='store_true', help='Include metrics plots at the plate level and heatmaps. Does not include plate level plots by default')
    
    # get arguments from the command line
    args = parser.parse_args()
            
    generate_report(args.project, args.run, args.qc, args.location, args.include_plates)

