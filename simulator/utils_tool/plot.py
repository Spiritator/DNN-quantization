# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:59:00 2019

@author: Yung-Yu Tsai

Code for fault tolerance metrics plotting
"""

import os,csv,glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.ticker import StrMethodFormatter
from PIL import Image
import numpy as np

def _preprocess_float_fault_rate_text(fl_fr_text):
    if 'e-' in fl_fr_text:
        ed_id=fl_fr_text.find('e-')
        flfrnew=fl_fr_text[0]+fl_fr_text[ed_id:]
    else:
        flfrnew=fl_fr_text
    
    return flfrnew

def make_FT_report(stat_dir,report_csv_filename=None):
    """
    Organize multiple scheme run result csv files into one report

    Parameters
    ----------
    stat_dir : String
        The directory contains multile result files. 
        Each of them stores multiple fault generation runs under one fault condition configuration.
        The condition is describe by filename.
    report_csv_filename : String. Don't need to contain '.csv' file extension in this argument.
        The filename for report csv file. The default is None.
        If type is String, write the combined analysis result into csv report file. 
        If None, don't write file, just return data statistic dictionary.

    Returns
    -------
    stat_data : Dictionary
        | Data structure
        | { experiment_variable_1 : { metric1 : { statistic1 : value,
        |                                         statistic2 : value,
        |                                         ... },
        |                             metric2 : { statistic1 : value,
        |                                         statistic2 : value,
        |                                         ... },
        |                             ...},
        |   experiment_variable_2 : { metric1 : { statistic1 : value,
        |                                         statistic2 : value,
        |                                         ... },
        |                             metric2 : { statistic1 : value,
        |                                         statistic2 : value,
        |                                         ... },
        |                             ...},
        |   ...}

    """
    stat_file_list=os.listdir(stat_dir)
    stat_files=dict()
    for fname in stat_file_list:
        if fname.endswith('.csv'):
            stat,f_ext=os.path.splitext(fname)
            stat=_preprocess_float_fault_rate_text(stat)
            stat_files[float(stat)]=fname
        #TODO
        # not just float number as variable
        
    stat_data=dict()
    for key in sorted(stat_files.keys()):
        with open(os.path.join(stat_dir,stat_files[key]), 'r', newline='') as stat_csvfile:
            csvdata=csv.DictReader(stat_csvfile)
            analyzed_metrics=dict()
            metrics=csvdata.fieldnames
            for metric in metrics:
                analyzed_metrics[metric]=np.array([],dtype=np.float32)

            for row in csvdata:
                for metric in metrics:
                    analyzed_metrics[metric]=np.append(analyzed_metrics[metric],float(row[metric]))
            
            stat_data[key]=analyzed_metrics
    
    #analysis
    for key in stat_data.keys():
        for keyy in stat_data[key].keys():
            analyzed_metrics=dict()
            metric_arr=stat_data[key][keyy]
            #TODO
            # prevent invalid data type to analysis
            analyzed_metrics['avg']=np.mean(metric_arr)
            analyzed_metrics['std_dev']=np.std(metric_arr)
            analyzed_metrics['max']=np.max(metric_arr)
            analyzed_metrics['min']=np.min(metric_arr)
            analyzed_metrics['var_up']=np.clip(analyzed_metrics['avg']+analyzed_metrics['std_dev'],0,analyzed_metrics['max'])
            analyzed_metrics['var_down']=np.clip(analyzed_metrics['avg']-analyzed_metrics['std_dev'],analyzed_metrics['min'],np.inf)
            stat_data[key][keyy]=analyzed_metrics
            
    if isinstance(report_csv_filename,str):
        repo_dir=os.path.split(stat_dir)
        repo_dir=repo_dir[0]
        
        with open(os.path.join(repo_dir,report_csv_filename+'.csv'), 'w', newline='') as repo_csvfile:
            for key in stat_data.keys():
                report_fieldnames=[key]+metrics
                writer=csv.DictWriter(repo_csvfile, fieldnames=report_fieldnames)
                writer.writeheader()
                for analysis in ['avg','std_dev','max','min','var_up','var_down']:
                    analysis_result_dict=dict()
                    analysis_result_dict[key]=analysis
                    for keyy in stat_data[key].keys():
                        analysis_result_dict[keyy]=stat_data[key][keyy][analysis]
                    writer.writerow(analysis_result_dict)

            
    return stat_data

def plot_FT_analysis(stat_dir=None,report_filename=None,font_size=None,legend_size=None,save_plot_format='png',dpi=250):
    """
    Make the fault tolerance report into line chart with statistic result

    Parameters
    ----------
    stat_dir : String, optional
        The directory contain multiple scheme run result csv files, which are made into one report. The default is None.
    report_filename : String, optional
        The directory to organized report csv file. The default is None.
    font_size : Integer, optional
        Font size of figure. The default is None.
    legend_size : Integer, optional
        Size of plot legend. The default is None.
    save_plot_format : String, optional. one of 'png', 'jpg', 'eps'
        The image format of plot saving. The default is 'png'.
    dpi: Integer
        The image defination (digit per inch).

    Returns
    -------
    stat_data : Dictionary
        | Data structure
        | { experiment_variable_1 : { metric1 : { statistic1 : value,
        |                                         statistic2 : value,
        |                                         ... },
        |                             metric2 : { statistic1 : value,
        |                                         statistic2 : value,
        |                                         ... },
        |                             ...},
        |   experiment_variable_2 : { metric1 : { statistic1 : value,
        |                                         statistic2 : value,
        |                                         ... },
        |                             metric2 : { statistic1 : value,
        |                                         statistic2 : value,
        |                                         ... },
        |                             ...},
        |   ...}

    """
    mpl.rcParams['figure.dpi']=dpi
    
    if stat_dir is None and report_filename is None:
        raise ValueError('Both argument stat_dir and report_filename are None! Choose one of them as data to draw analysis plot.')
        
    if stat_dir is None:
        with open(os.path.join(report_filename), 'r', newline='') as repo_csvfile:
            csvdata=csv.DictReader(repo_csvfile)
            stat_data=dict()
            sub_stat=dict()
            metrics=csvdata.fieldnames
            stat, metrics= metrics[0],metrics[1:]
            for metric in metrics:
                sub_stat[metric]=dict()
                
            stat_tmp=float(stat)
            for row in csvdata:
                if row[stat] in ['avg','std_dev','max','min','var_up','var_down']:
                    for metric in metrics:
                        sub_stat[metric][row[stat]]=float(row[metric])
                else:
                    stat_data[stat_tmp]=sub_stat
                    sub_stat=dict()
                    for metric in metrics:
                        sub_stat[metric]=dict()
                    stat_tmp=float(row[stat])
                    
            stat_data[stat_tmp]=sub_stat
            
    if report_filename is None:
        stat_data=make_FT_report(stat_dir)
        
        
    x=list(stat_data.keys())
    metrics=list(stat_data[x[0]].keys())
    
    if save_plot_format not in ['png','eps']:
        raise ValueError('Plot save file must be either png or eps format.')

    for metric in metrics:
        fig = plt.figure()
        
        if font_size is not None:
            plt.rcParams.update({'font.size': font_size})
        
        avg=[stat_data[xid][metric]['avg'] for xid in x]

        maxx=[stat_data[xid][metric]['max'] for xid in x]
        plt.plot(x, maxx, label='max', c='dodgerblue', linestyle='--', marker='.')
        
        minn=[stat_data[xid][metric]['min'] for xid in x]
        plt.plot(x, minn, label='min', c='dodgerblue', linestyle='-.', marker='.')
        
        var_up=[stat_data[xid][metric]['var_up'] for xid in x]
        var_down=[stat_data[xid][metric]['var_down'] for xid in x]
        
        plt.fill_between(x, var_up, var_down, facecolor = 'lightgray', label='variance')
        
        plt.plot(x, avg, label='average', c='darkblue', marker='.')
            
        plt.title(metric)
        plt.ylabel(metric)
        plt.xlabel('bit fault rate')
        plt.xscale('log')
        
        if legend_size is None:
            if font_size is None:
                legend_size=10
            else:
                legend_size=font_size
        
        if 'loss' in metric:
            plt.legend(loc='lower right',prop={'size': legend_size})
        elif 'acc' in metric:
            plt.legend(loc='upper right',prop={'size': legend_size})
        else:
            plt.legend(loc='lower right',prop={'size': legend_size})
        plt.tight_layout()
        if stat_dir is not None:
            pic_path=stat_dir+'/'+metric+'.'+save_plot_format
        elif report_filename is not None:
            pic_path=os.path.split(report_filename)[0]+'/'+metric+'.'+save_plot_format
            
        if save_plot_format=='eps':
            plt.savefig(pic_path, format='eps')
        else:
            plt.savefig(pic_path,dpi=dpi)
    
    return stat_data

def plot_FT_analysis_multiple(stat_data_list,plot_save_dir,plot_color_list,label_list,font_size=None,legend_size=None,save_plot_format='png',dpi=250):
    """
    Make multiple fault tolerance report into one line chart with statistic result.
    Each metric plot a figure contains multiple report sources.

    Parameters
    ----------
    stat_data_list : List of Dictionaries
        List of stat_data Dictionaries that are being plotted. The list order will be followed by later parameters.
        
        stat_data : Dictionary
            | Data structure
            | { experiment_variable_1 : { metric1 : { statistic1 : value,
            |                                         statistic2 : value,
            |                                         ... },
            |                             metric2 : { statistic1 : value,
            |                                         statistic2 : value,
            |                                         ... },
            |                             ...},
            |   experiment_variable_2 : { metric1 : { statistic1 : value,
            |                                         statistic2 : value,
            |                                         ... },
            |                             metric2 : { statistic1 : value,
            |                                         statistic2 : value,
            |                                         ... },
            |                             ...},
            |   ...}
        
    plot_save_dir : String
        The directory where plots are going to save.
    plot_color_list : List of Dictionaries
        The colors of plot lines. The List order must follow the stat_data_list. Each Dictionary should be in format:
            
            | plot_color: Dictionary format 
            | {'max':'color_of_max_line',
            |  'min':'color_of_min_line',
            |  'avg':'color_of_avg_line',
            |  'var':'color_of_var_line'}. 
            The dictionary values are string of the matplotlib.pyplot color scheme.

            
    label_list : List of String
        The line label. The List order must follow the stat_data_list.
    font_size : Integer, optional
        Font size of figure. The default is None.
    legend_size : Integer, optional
        Size of plot legend. The default is None.
    save_plot_format : String, optional. one of 'png', 'jpg', 'eps'
        The image format of plot saving. The default is 'png'.
    dpi: Integer
        The image defination (digit per inch).

    Returns
    -------
    None
        Plot multiple report sources fault tolerance report

    """
    mpl.rcParams['figure.dpi']=dpi
    
    if not isinstance(stat_data_list,list):
        raise TypeError('argument stat_data_list should be type list consist of dictionary of stat_data of a FT analysis.')
    if not isinstance(plot_color_list,list):
        raise TypeError('argument plot_color_list should be type list consist of dictionary of color sheme of a FT analysis in pyplot color format string.')
    if len(stat_data_list)!=len(plot_color_list):
        raise ValueError('stat_data_list not equal to plot_color_list please check your data.')        
        
    x=list(stat_data_list[0].keys())
    metrics=list(stat_data_list[0][x[0]].keys())

    if save_plot_format not in ['png','eps']:
        raise ValueError('Plot save file must be either png or eps format.')

    for iterr,metric in enumerate(metrics):
        fig = plt.figure()
        
        if font_size is not None:
            plt.rcParams.update({'font.size': font_size})

        
        for i in range(len(stat_data_list)):
            x=list(stat_data_list[i].keys())
            metrics_i=list(stat_data_list[i][x[0]].keys())
            
            avg=[stat_data_list[i][xid][metrics_i[iterr]]['avg'] for xid in x]
    
            maxx=[stat_data_list[i][xid][metrics_i[iterr]]['max'] for xid in x]
            plt.plot(x, maxx, c=plot_color_list[i]['max'], linestyle='-', marker='^')
            
            minn=[stat_data_list[i][xid][metrics_i[iterr]]['min'] for xid in x]
            plt.plot(x, minn, c=plot_color_list[i]['min'], linestyle='-', marker='v')
            
            var_up=[stat_data_list[i][xid][metrics_i[iterr]]['var_up'] for xid in x]
            var_down=[stat_data_list[i][xid][metrics_i[iterr]]['var_down'] for xid in x]
            
            plt.fill_between(x, var_up, var_down, facecolor = plot_color_list[i]['var'], alpha=0.5)
            
            plt.plot(x, avg, label=label_list[i], c=plot_color_list[i]['avg'], marker='.')
            
        plt.title(metric)
        plt.ylabel(metric)
        plt.xlabel('bit fault rate')
        plt.xscale('log')
        
        if legend_size is None:
            if font_size is None:
                legend_size=10
            else:
                legend_size=font_size
       
        if 'loss' in metric:
            plt.legend(loc='lower right',prop={'size': legend_size})
        elif 'acc' in metric:
            plt.legend(loc='upper right',prop={'size': legend_size})
        else:
            plt.legend(loc='lower right',prop={'size': legend_size})
        plt.tight_layout()
        pic_path=plot_save_dir+'/'+metric+'.'+save_plot_format
        
        if save_plot_format=='eps':
            plt.savefig(pic_path, format='eps')
        else:
            plt.savefig(pic_path,dpi=dpi)
    
def _heatmap(data, row_labels, col_labels,
             ax=None, cbar_kw={}, cbarlabel="", 
             aspect_ratio=0.4, xtick_rot=0, label_redu=None, grid_width=3, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, aspect=aspect_ratio, **kwargs)
            
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    if label_redu is not None:
        # reduced amount of tick label for better visualization on large heatmap
        ax.set_xticklabels([x if i%label_redu==0 else ' ' for i,x in enumerate(col_labels)])
        ax.set_yticklabels([x if i%label_redu==0 else ' ' for i,x in enumerate(row_labels)])
    else:
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=xtick_rot, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=grid_width)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def _annotate_heatmap(im, data=None, text=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, vmax=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    text
        Text used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    vmax
        Value in data unit set as the max of color bar.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    if text is None:
        dataprint=data
    else:
        dataprint=text

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = [0.0,1.0]
        threshold[0] = im.norm(data.max())*0.2
        threshold[1] = im.norm(data.max())*0.8

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold[1] or im.norm(data[i, j]) < threshold[0])])
            text = im.axes.text(j, i, valfmt(dataprint[i, j], None), **kw)
            texts.append(text)

    return texts

def plot_FT_2D_heatmap(stat_data_dict, plot_save_dir, row_labels, col_labels, 
                       xlabel, ylabel, sparse=False, aspect_ratio=0.3, 
                       vmax_dict=None, annotate=True, text_data=None, valfmt="{x:.3f}", 
                       xtick_rot=0, label_redu=None, grid_width=3, text_size=12, cbar_ref_level=0,
                       save_plot_format='png',dpi=250):
    """
    Plot fault tolerance report in 2D heatmap. For the data report contain 2 experiment variables.
    Each experiment variable represent in a dimension of plot.
    The value of fault tolerance metrics are showing in block color.

    Parameters
    ----------
    stat_data_dict : Dictionary
        The dictionary contain a stat_data in each item. The key and value pair is the x axis of plot.
        The another experiment variable in each stat_data is the y axis of plot.
        The format of Dictionary is:
            | { fault_tolerance_metric_1 : { metric_statistics_1 : [[2D_array_data]],
            |                                metric_statistics_1 : [[2D_array_data]],
            |                                ...},
            |   fault_tolerance_metric_2 : { metric_statistics_1 : [[2D_array_data]],
            |                                metric_statistics_1 : [[2D_array_data]],
            |                                ...},
            |   ...}
            
    plot_save_dir : String
        The directory where plots are going to save.
    row_labels : List or Ndarray
        The labels for the rows of plot.
    col_labels : List or Ndarray
        The labels for the columns of plot.
    xlabel : String
        The name of x axis.
    ylabel : String
        The name of y axis
    sparse: Bool
        Plot sparse heatmap or not.
    aspect_ratio : Float, optional
        The aspect ratio of heatmap. The default is 0.3.
    vmax_dict: Dictionary
        The maximun for color reference need one for each heatmap.
        The dictionary structure and max value must align with stat_data_dict.
        The format of Dictionary is:
            | text_dict[fault_bit][metric][vmax]
    valfmt : String, optional
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`. The default is "{x:.3f}".
    annotate : Bool, optional
        Show the annotaion in heatmap or not. For heatmap with huge amount of blocks, 
        annotate may set to False for prevent chaotic layout of plot. 
        Let the metric value only represent by color. The default is True.
    text_data : Dictionary.
        The text are being append to next line of a data point.
        The dictionary structure and array of text must align with stat_data_dict.
        The format of Dictionary is:
            | text_dict[FT_metric][ 1D_List_Tuple:[coor...], 1D_array:[text...] ]
    xtick_rot : Float, optional
        Rotate the xtick label text, for preventing text overlap. The default is 0.
    label_redu : Integer, optional
        Reduce the precence of xtick and ytick labels, for preventing chaotic plot layout and text overlap.
        The label_redu value the interval of each label precence. The default is None.
    grid_width : Float, optional
        The width of heatmap block, for adjust the visual presentation. The default is 3.
    text_size: Integer. default 12
        The text font size of pixel annotation.
    cbar_ref_level: Integer. default 0
        The metric for colorbar reference. cbar_ref_level is the level of stat_data_dict keys that are being set as refernce.
    save_plot_format : String, optional. one of 'png', 'jpg', 'eps'
        The image format of plot saving. The default is 'png'.
    dpi: Integer
        The image defination (digit per inch).

    Returns
    -------
    None
        Plot multiple report sources fault tolerance report

    """
    mpl.rcParams['figure.dpi']=dpi

    if not os.path.isdir(plot_save_dir+'/plot'):
        os.mkdir(plot_save_dir+'/plot')
    
    if save_plot_format not in ['png','eps']:
        raise ValueError('Plot save file must be either png or eps format.')
    
    for mtrc in stat_data_dict.keys():
        for mtrcstat in stat_data_dict[mtrc].keys():
            FT_arr=stat_data_dict[mtrc][mtrcstat]
            
            fig, ax = plt.subplots()
            
            if cbar_ref_level==0:
                cbarref=mtrc
            else:
                cbarref=mtrcstat
                
            if 'acc' in cbarref and 'loss' not in cbarref:   
                if mtrcstat=='std_dev':
                    colorbar='RdYlGn_r'
                else:
                    colorbar='RdYlGn'
            else:
                colorbar='RdYlGn_r'
            
            if sparse:
                colorbar=sparse_colormap(colorbar)
                
            if vmax_dict is None:
                maxmtrc=None
            else:
                maxmtrc=vmax_dict[mtrc][mtrcstat]
                
            im, cbar = _heatmap(FT_arr, row_labels, col_labels,
                                ax=ax, cmap=colorbar, cbarlabel=cbarref, 
                                aspect_ratio=aspect_ratio, vmax=maxmtrc, xtick_rot=xtick_rot, 
                                label_redu=label_redu, grid_width=grid_width)
            
            if annotate:
                if text_data is None:
                    texts = _annotate_heatmap(im, valfmt=valfmt, fontsize=text_size)
                else:
                    textarr=text_data[mtrc]
                    if maxmtrc>99:
                        imfmt='{x:.2E}'
                    else:
                        imfmt=valfmt
                    imtext,imfmt=concate_value2text(FT_arr,textarr,imfmt)
                    texts = _annotate_heatmap(im, text=imtext, valfmt=imfmt, fontsize=text_size, threshold=[maxmtrc*0.2,maxmtrc*0.8])
            else:
                if text_data is not None:
                    textarr=text_data[mtrc]
                    imtext,imfmt=concate_value2text(FT_arr.shape,textarr,valfmt)
                    texts = _annotate_heatmap(im, text=imtext, valfmt=imfmt, fontsize=text_size, threshold=[maxmtrc*0.2,maxmtrc*0.8])
            
            plt.title(mtrc+'  '+mtrcstat)
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            fig.tight_layout()
            
            if not os.path.isdir(plot_save_dir+'/plot/'+mtrcstat):
                os.mkdir(plot_save_dir+'/plot/'+mtrcstat)
            
            pic_path=plot_save_dir+'/plot/'+mtrcstat+'/'+mtrc+'-'+mtrcstat+'.'+save_plot_format
            
            if save_plot_format=='eps':
                plt.savefig(pic_path, format='eps',bbox_inches='tight')
            else:
                plt.savefig(pic_path,dpi=dpi,bbox_inches='tight')
                
            plt.show()
            # pevent memory spikes
            plt.clf()

def concate_value2text(value,text,fmt_orig):
    if not isinstance(value,tuple):
        concattext=np.empty(value.shape, dtype=str).astype(np.object)
        for i,idx in enumerate(text['coor_list']):
            concattext[idx]=fmt_orig.format(x=value[idx])+'\n'+text['param'][i]
    else:
        concattext=np.empty(value, dtype=str).astype(np.object)
        for i,idx in enumerate(text['coor_list']):
            concattext[idx]=text['param'][i]
    fmt_new='{x}'
    return concattext,fmt_new

def sparse_colormap(cmap):
    newcolors= cm.get_cmap(cmap, 256)
    newcolors = newcolors(np.linspace(0, 1, 256))
    blank = np.array([1.0, 1.0, 1.0, 0.0])
    newcolors[0] = blank
    newcmp = ListedColormap(newcolors)
    return newcmp
    
def dict_format_lfms_to_ms2Dlf(stat_data_dict):
    """ 
    Convert FT data dictionay from format: 
        data_dict[layer][fault_rate][FT_metric][metric_statistics]
    to format:
        data_dict[FT_metric][metric_statistics][ 2D_array[ fault_rate : layer ] ]
    """
    var_dir_list=list(stat_data_dict.keys())
    # data transformation
    metric_list=list(stat_data_dict[var_dir_list[0]][0.1].keys())
    metric_stats=list(stat_data_dict[var_dir_list[0]][0.1]['loss'].keys())
    stat_data_metric_dict=dict()
    
    statvd_len=[len(stat_data_dict[varr]) for varr in var_dir_list] # get biggest data layer
    var_len=max(statvd_len)
    argvarlen=statvd_len.index(var_len)
    
    fr_list=list(stat_data_dict[var_dir_list[argvarlen]].keys())
    
    for mtrc in metric_list:
        sdmd_tmp=dict()
        for mtrcstat in metric_stats:
            if 'acc' in mtrc and 'loss' not in mtrc:
                if mtrcstat=='std_dev':
                    data_tmp=np.zeros((var_len,len(var_dir_list)))
                else:
                    data_tmp=np.ones((var_len,len(var_dir_list)),dtype=float)
            else:
                data_tmp=np.zeros((var_len,len(var_dir_list)))
            for i,layer in enumerate(var_dir_list):
                for j,fr in enumerate(fr_list):
                    if fr in stat_data_dict[layer].keys():
                        data_tmp[j,i]=stat_data_dict[layer][fr][mtrc][mtrcstat]
            sdmd_tmp[mtrcstat]=data_tmp
        stat_data_metric_dict[mtrc]=sdmd_tmp
        
    return stat_data_metric_dict,fr_list


def collect_metric_PE(file_name):
    """ Collect PE array fault simulation result from file.

    Parameters
    ----------
    file_name : String
        The directory to PE array fault simulation result file.

    Returns
    -------
    report : Dictionary
        The collected metrics keys are metric name, values are 1D array of metric data from each iteration.
    """
    with open(file_name, 'r', newline='') as result_file:
        csvdata=csv.DictReader(result_file)
        metrics=csvdata.fieldnames
        
        report=dict()
        for metric in metrics:
            if metric in ['PE y','PE x','SA bit','num psidx']:
                report[metric]=np.array([],dtype=np.int32)
            elif metric in ['param','SA type']:
                report[metric]=np.array([],dtype=np.str)
            else:
                report[metric]=np.array([],dtype=np.float32)
        for row in csvdata:
            for metric in metrics:
                if metric in ['PE y','PE x','SA bit','num psidx']:
                    report[metric]=np.append(report[metric],np.int32(row[metric]))
                elif metric in ['param','SA type']:
                    report[metric]=np.append(report[metric],np.str(row[metric]))
                else:
                    report[metric]=np.append(report[metric],np.float32(row[metric]))
        
    return report
            
def dict_format_mfv_to_b2Dm(data_dict,n_PEy,n_PEx,datatextslice=None):
    """ 
    Convert FT data dictionay from format: 
        data_dict[metric & fault_info][value]
    to format:
        data_dict[fault_bit][metric][ 2D_array[ PE_y : PE_x ] ]
        text_dict[FT_metric][ 1D_List_Tuple:[coor...], 1D_array:[text...] ]
    """
    sorter=np.argsort(data_dict['SA bit'])
    SA_bits,cnt_idx=np.unique(data_dict['SA bit'],return_counts=True)
    cnt_idx=np.cumsum(cnt_idx)[:-1]
    vmax_dict=dict()
    for metric in data_dict:
        if metric not in ['PE y','PE x','SA bit','param','SA type']:
            vmax_dict[metric]=np.max(data_dict[metric])
        elif metric=='param' and datatextslice is not None:
            for i in range(len(data_dict[metric])):
                data_dict[metric][i]=data_dict[metric][i][datatextslice]
        data_dict[metric]=np.split(data_dict[metric][sorter],cnt_idx)
    # [metric][fault_bit][value]            
    new_data_dict=dict()
    new_text_dict=dict()
    new_vmax_dict=dict()
    if np.max(SA_bits)<10:
        formater='bit-%d'
    else:
        formater='bit-%2d'
    for bit in SA_bits:
        metric_tmp=dict()
        vmax_tmp=dict()
        for metric in data_dict:
            metric_tmp[metric]=data_dict[metric][bit]
            if metric not in ['PE y','PE x','SA bit','param','SA type']:
                vmax_tmp[metric]=vmax_dict[metric]
        new_vmax_dict[formater%bit]=vmax_tmp
        new_data_dict[formater%bit]=metric_tmp
        new_text_dict[formater%bit]=dict()
    # [fault_bit][metric][value]
    for segment in new_data_dict:
        heatframe=dict()
        coorlist=list(zip(new_data_dict[segment]['PE y'],new_data_dict[segment]['PE x']))
        new_text_dict[segment]['coor_list']=coorlist
        new_text_dict[segment]['param']=new_data_dict[segment]['param']
        #heatframe['SA type']=new_data_dict[segment]['SA type']
        for metric in new_data_dict[segment]:
            if metric in ['PE y','PE x','SA bit','param','SA type']:
                continue
            maptmp=np.zeros([n_PEy,n_PEx],new_data_dict[segment][metric].dtype)
            for i,coor in enumerate(coorlist):
                maptmp[coor]=new_data_dict[segment][metric][i]
            heatframe[metric]=maptmp
            
        new_data_dict[segment]=heatframe
    #[fault_bit][metric][ 2D_array[ PE_y : PE_x ] ]
    return new_data_dict, new_text_dict, new_vmax_dict


def make_GIF(filepath,save_GIF_name,file_ext=None,duration=200,loop=0):
    """ Collect images from given file path and cram images into GIF motion picture

    Parameters
    ----------
    filepath : String
        The file path to images. Could include the common part of filenames start.
    save_GIF_name : String
        The filename for output GIF under filepath.
    file_ext : String, optional
        File extention of input filepath images. If None, collect all files under filepath.
    duration : Integer
        The duration of each images in ms. The default is 200.
    loop : Integer, optional
        Number of times the GIF should loop. 0 means that it will loop forever.. The default is 0.

    """
    if file_ext is None:
        fp_in = filepath+'*'
    else:
        fp_in = filepath+'*.'+file_ext
    fp_out = filepath+save_GIF_name+'.gif'
    
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=duration, loop=loop)


