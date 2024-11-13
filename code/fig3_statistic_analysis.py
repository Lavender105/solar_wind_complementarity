# -*- coding: utf-8 -*
import os
import argparse
import rasterio
import cfgrib
import random
import json
import math
import multiprocessing
import xarray as xr
import numpy as np
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import time
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Circle, Arc, PathPatch
from matplotlib.path import Path
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as mticker
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.lines as mlines
from matplotlib.colors import to_rgba

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from cartopy.mpl.patch import geos_to_path
import shapely.geometry as sgeom
from copy import copy

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Helvetica'
plt.rcParams['mathtext.it'] = 'Helvetica:italic'
plt.rcParams['mathtext.bf'] = 'Helvetica:bold'


def find_side(ls, side):
    """
 Given a shapely LineString which is assumed to be rectangular, return the
 line corresponding to a given side of the rectangle.

 """
    minx, miny, maxx, maxy = ls.bounds
    points = {'left': [(minx, miny), (minx, maxy)],
              'right': [(maxx, miny), (maxx, maxy)],
              'bottom': [(minx, miny), (maxx, miny)],
              'top': [(minx, maxy), (maxx, maxy)],}
    return sgeom.LineString(points[side])

def lambert_xticks(ax, ticks, side='bottom', direction='out', length=3):
    """Draw ticks on the bottom x-axis of a Lambert Conformal projection."""
    te = lambda xy: xy[0]
    lc = lambda t, n, b: np.vstack((np.zeros(n) + t, np.linspace(b[2], b[3], n))).T
    xticks, xticklabels = _lambert_ticks(ax, ticks, side, lc, te)
    if side == 'bottom':
        ax.xaxis.tick_bottom()
    else:
        ax.xaxis.tick_top()
    ax.tick_params(axis='x', width=0.5, direction=direction, length=length)
    ax.set_xticks(xticks)
    # ax.set_xticklabels([ax.xaxis.get_major_formatter()(xtick) for xtick in xticklabels])
    ax.set_xticklabels([])

def lambert_yticks(ax, ticks, side='left', direction='out', length=3):
    """Draw ricks on the left y-axis of a Lamber Conformal projection."""
    te = lambda xy: xy[1]
    lc = lambda t, n, b: np.vstack((np.linspace(b[0], b[1], n), np.zeros(n) + t)).T
    yticks, yticklabels = _lambert_ticks(ax, ticks, side, lc, te)
    if side == 'left':
        ax.yaxis.tick_left()
    else:
        ax.yaxis.tick_right()
    ax.tick_params(axis='y', width=0.5, direction=direction, length=length)
    ax.set_yticks(yticks)
    # ax.set_yticklabels([ax.yaxis.get_major_formatter()(ytick) for ytick in yticklabels])
    ax.set_yticklabels([])

def _lambert_ticks(ax, ticks, tick_location, line_constructor, tick_extractor):
    """Get the tick locations and labels for an axis of a Lambert Conformal projection."""
    outline_patch = sgeom.LineString(ax.spines['geo'].get_path().vertices.tolist())
    axis = find_side(outline_patch, tick_location)
    n_steps = 30
    extent = ax.get_extent(ccrs.PlateCarree())
    _ticks = []
    for t in ticks:
        xy = line_constructor(t, n_steps, extent)
        proj_xyz = ax.projection.transform_points(ccrs.Geodetic(), xy[:, 0], xy[:, 1])
        xyt = proj_xyz[..., :2]
        ls = sgeom.LineString(xyt.tolist())
        locs = axis.intersection(ls)
        if not locs:
            tick = [None]
        else:
            tick = tick_extractor(locs.xy)
        _ticks.append(tick[0])
    # Remove ticks that aren't visible: 
    ticklabels = copy(ticks)
    while True:
        try:
            index = _ticks.index(None)
        except ValueError:
            break
        _ticks.pop(index)
        ticklabels.pop(index)
    return _ticks, ticklabels

def min_value_and_index(row):
    min_value = row.min()
    min_index = row.argmin()
    return pd.Series([min_value, min_index])

def draw_cv_map(gdf, column, min_val, max_val, colormap, legend_label, cbar_label, save_path, alpha=1, cbar_ticks=[], twoslopenorm=False):
    font_path = '/System/Library/Fonts/Helvetica.ttc'
    prop = FontProperties(fname=font_path)

    jiuduanxian_path = f'../中国区划shp/中国区划-权威/九段线.shp'
    jiuduanxian = gpd.read_file(jiuduanxian_path)

    projection = ccrs.LambertConformal(central_longitude=105, central_latitude=35, standard_parallels=(30, 60))
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': projection})
    ax.set_extent([76, 132, 16, 53.5], ccrs.PlateCarree())
    ax_n = fig.add_axes([0.78, 0.23, 0.1, 0.12], projection = projection)
    ax_n.set_extent([104.5, 125, 0, 26])
    fig.canvas.draw()
    ax.add_geometries(jiuduanxian["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black", linewidth=.3)

    xticks = [60, 80, 100, 120, 140]
    yticks = [20, 30, 40, 50]
    gl = ax.gridlines(xlocs=xticks, ylocs=yticks, draw_labels=True, x_inline=False, y_inline=False, rotate_labels=0, linestyle='--', linewidth=0.5)
    # ax.gridlines(xlocs=xticks, ylocs=yticks, linestyle='--', linewidth=0.5)
    gl.xlabel_style = {'fontproperties': prop, 'size': 14}
    gl.ylabel_style = {'fontproperties': prop, 'size': 14}
    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    lambert_xticks(ax, xticks)
    lambert_yticks(ax, yticks)

    ax_n.add_geometries(jiuduanxian["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black", linewidth=.3)
    # fig.canvas.draw()
    xticks = [110, 120]
    yticks = [10, 20]
    gl = ax_n.gridlines(draw_labels=True, x_inline=False, y_inline=False, rotate_labels=0, linewidth=0.1, linestyle='--')
    gl.top_labels = False
    gl.left_labels = False
    gl.right_labels = False
    gl.bottom_labels = False


    cmap = get_cmap(colormap)
    if alpha < 1:
        colors = cmap(np.arange(cmap.N))
        colors[:, -1] = alpha
        cmap = ListedColormap(colors)
    # norm = Normalize(vmin=gdf[column].min(), vmax=gdf[column].max())
    if twoslopenorm:
        norm = TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)
    else:
        norm = Normalize(vmin=min_val, vmax=max_val)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    t1 = time.time()
    for idx, row in gdf.iterrows():
        if row['geometry'].geom_type in ['Polygon', 'MultiPolygon'] and row['geometry'].area < 1e-6:
            continue
        color = sm.to_rgba(row[column]) if row[column] != -1 else '0.95' # 获取数据对应的颜色
        ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor=color, edgecolor='black', linewidth=0.1)
        ax_n.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor=color, edgecolor='black', linewidth=0.03)
    t2 = time.time()
    print(f'complete drawing correlation, spend {t2 - t1} seconds.')

    legend_list = [mpatches.Patch(facecolor='0.95', edgecolor='0.2', linewidth=0.2, label=legend_label)]

    sm.set_array([])
    vmin = min_val # gdf[column].min()
    vmax = max_val # gdf[column].max()
    vcenter = round((vmin + vmax) / 2, 2)
    sm.set_clim(vmin=vmin, vmax=vmax)
    axins = ax.inset_axes([0.85, 0.32, 0.03, 0.33]) # (x, y, width, height)
    cbar = plt.colorbar(sm, cax=axins, label=cbar_label)
    cbar.set_label(cbar_label, fontsize=14, fontproperties=prop)
    if cbar_ticks == []:
        cbar_ticks = [vmin, vcenter, vmax]
        cbar_ticklabels = [f'{vmin:.2f}', f'{vcenter:.2f}', f'{vmax:.2f}']
    else:
        # cbar_ticklabels = [f'{v:.1f}' for v in cbar_ticks]
        cbar_ticklabels = [f'{v}' for v in cbar_ticks]
    cbar.set_ticks(cbar_ticks)
    cbar.ax.set_yticklabels(cbar_ticklabels, fontproperties=prop)
    cbar.ax.tick_params(length=2, width=0.5, labelsize=14)
    cbar.outline.set_linewidth(0.5)
    leg = ax.legend(handles=legend_list, bbox_to_anchor=(0.005, 0.01), loc='lower left', fontsize=14, title='', shadow=False, fancybox=False, framealpha=0)#, prop=prop)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    t3 = time.time()
    print(f'save map to {save_path}. spend {t3 - t2} seconds.')


def calculate_metrics(x):
    if len(x) == 8760:
        array = np.array(x)
        mean = np.mean(array)
        variance = np.var(array)
        std_dev = np.std(array)
        mad = np.mean(np.abs(array - np.mean(array)))
        cv = std_dev / mean
        return pd.Series([mean, variance, std_dev, mad, cv])
    else:
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])

def scatter_plot_solar(before, after, save_path):
    font_path = '/System/Library/Fonts/Helvetica.ttc'
    prop = FontProperties(fname=font_path, size=14)

    diff = before - after
    fig, ax = plt.subplots(figsize=(5.8, 6))
    scatter = ax.scatter(before, after, c=diff, cmap='coolwarm', s=np.abs(diff)*50, alpha=0.3)
    
    min_val = min(min(before), min(after))
    max_val = max(max(before), max(after))
    ax.plot([min_val, max_val], [min_val, max_val], color='0.2', linestyle='--', linewidth=1, alpha=0.7, label='No change line')
    
    x_ticks = np.arange(np.floor(min_val*10)/10, np.ceil(max_val*10)/10, 0.2)
    y_ticks = np.arange(np.floor(min_val*10)/10, np.ceil(max_val*10)/10, 0.2)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(prop)
        label.set_fontsize(14)

    diff_min = np.min(diff)
    diff_max = np.max(diff)
    norm_labels = [(x - diff_min) / (diff_max - diff_min) for x in [0.6, 0.4, 0.2, 0.1]]
    marker_sizes = [np.sqrt(np.abs(x) * 50) for x in [0.6, 0.4, 0.2, 0.1]]
    handles = [
        mlines.Line2D([], [], marker='o', markeredgecolor=plt.cm.coolwarm(norm_labels[0]), markerfacecolor=to_rgba(plt.cm.coolwarm(norm_labels[0]), alpha=0.3), markersize=marker_sizes[0], linestyle='None', label='0.6'),
        mlines.Line2D([], [], marker='o', markeredgecolor=plt.cm.coolwarm(norm_labels[1]), markerfacecolor=to_rgba(plt.cm.coolwarm(norm_labels[1]), alpha=0.3), markersize=marker_sizes[1], linestyle='None', label='0.4'),
        mlines.Line2D([], [], marker='o', markeredgecolor=plt.cm.coolwarm(norm_labels[2]), markerfacecolor=to_rgba(plt.cm.coolwarm(norm_labels[2]), alpha=0.3), markersize=marker_sizes[2], linestyle='None', label='0.2'),
        mlines.Line2D([], [], marker='o', markeredgecolor=plt.cm.coolwarm(norm_labels[3]), markerfacecolor=to_rgba(plt.cm.coolwarm(norm_labels[3]), alpha=0.3), markersize=marker_sizes[3], linestyle='None', label='0.1') 
    ]
    leg = ax.legend(handles=handles, loc='lower left', fontsize=14, title=r'$\Delta$CV (before-after)', shadow=False, fancybox=False, framealpha=0.8, prop=prop)
    leg.set_title(r'$\Delta$CV (before-after)', prop=prop)
    leg.get_frame().set_linewidth(0)

    ax.set_xlabel('CV before complementarity', fontsize=14, fontproperties=prop)
    ax.set_ylabel('CV after complementarity', fontsize=14, fontproperties=prop)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved map to {save_path}')


def scatter_plot_wind(before, after, save_path):
    font_path = '/System/Library/Fonts/Helvetica.ttc'
    prop = FontProperties(fname=font_path, size=14)

    diff = before - after
    fig, ax = plt.subplots(figsize=(5.8, 6))

    colormap = 'coolwarm'
    norm = TwoSlopeNorm(vmin=diff.min(), vcenter=0, vmax=diff.max())
    scatter = ax.scatter(before, after, c=diff, cmap=colormap, norm=norm, s=np.abs(diff)*50, alpha=0.3)
    
    min_val = min(min(before), min(after))
    max_val = max(max(before), max(after))
    ax.plot([min_val, max_val], [min_val, max_val], color='0.2', linestyle='--', linewidth=1, alpha=0.7, label='No change line')
    
    x_ticks = np.arange(np.floor(min_val*10)/10, np.ceil(max_val*10)/10 + 0.1, 1)
    y_ticks = np.arange(np.floor(min_val*10)/10, np.ceil(max_val*10)/10 + 0.1, 1)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(prop)
        label.set_fontsize(14)

    diff_min = np.min(diff)
    diff_max = np.max(diff)
    norm_labels = [(x - diff_min) / (diff_max - diff_min) for x in [5, 3, 1, -1]]
    marker_sizes = [np.sqrt(np.abs(x) * 50) for x in [5, 3, 1, -1]]
    handles = [
        mlines.Line2D([], [], marker='o', markeredgecolor=plt.cm.coolwarm(norm_labels[0]), markerfacecolor=to_rgba(plt.cm.coolwarm(norm_labels[0]), alpha=0.3), markersize=marker_sizes[0], linestyle='None', label='5'),
        mlines.Line2D([], [], marker='o', markeredgecolor=plt.cm.coolwarm(norm_labels[1]), markerfacecolor=to_rgba(plt.cm.coolwarm(norm_labels[1]), alpha=0.3), markersize=marker_sizes[1], linestyle='None', label='3'),
        mlines.Line2D([], [], marker='o', markeredgecolor=plt.cm.coolwarm(norm_labels[2]), markerfacecolor=to_rgba(plt.cm.coolwarm(norm_labels[2]), alpha=0.3), markersize=marker_sizes[2], linestyle='None', label='1'),
        mlines.Line2D([], [], marker='o', markeredgecolor=plt.cm.coolwarm(norm_labels[3]), markerfacecolor=to_rgba(plt.cm.coolwarm(norm_labels[3]), alpha=0.3), markersize=marker_sizes[3], linestyle='None', label='-1') 
    ]
    leg = ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.55, 0.4),  fontsize=14, title=r'$\Delta$CV (before-after)', shadow=False, fancybox=False, framealpha=0.8, prop=prop)
    leg.set_title(r'$\Delta$CV (before-after)', prop=prop)
    leg.get_frame().set_linewidth(0)

    ax.set_xlabel('CV before complementarity', fontsize=14, fontproperties=prop)
    ax.set_ylabel('CV after complementarity', fontsize=14, fontproperties=prop)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved map to {save_path}')


def boxplot(data, save_path):
    font_path = '/System/Library/Fonts/Helvetica.ttc'
    prop = FontProperties(fname=font_path)

    df = pd.DataFrame(data)

    plt.figure(figsize=(4.8, 6))
    color_palette = ['#F8B072', '#EC3E31', '#A6D0E6', '#8582BD']
    ax = sns.boxplot(data=df, width=0.45, linewidth=0.5, showfliers=False, palette=color_palette)
    ax.grid(True, axis='y', linestyle='-', linewidth=0.3, color='gray')

    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=prop, fontsize=14)
    ax.set_yticklabels(ax.get_yticks(), fontproperties=prop, fontsize=14)

    # ax.text(0.4, 2.7, 'Coefficient of Variation', fontproperties=bold_prop, fontsize=10, ha='center', va='center')
    ax.set_xlabel('Category', fontsize=14, fontproperties=prop)
    ax.set_ylabel('CV', fontsize=14, fontproperties=prop)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved map to {save_path}')


def draw_violin_plot(data, save_path):
    font_path = '/System/Library/Fonts/Helvetica.ttc'
    prop = FontProperties(fname=font_path, size=14)

    df = pd.DataFrame(data)
    palette = {'Solar': '#F8B072', 'Wind': '#A6D0E6'}
    # palette = {'Solar': sns.color_palette("pastel")[1], 'Wind': sns.color_palette("pastel")[0]}
    # plt.figure(figsize=(16.8, 6)) # v1
    plt.figure(figsize=(7.8, 6))
    sns.violinplot(x='Level', y='Correlation', hue='Type', data=df, split=True, inner='quart', palette=palette, linewidth=0.5)
    position = 2
    plt.fill_betweenx([0, 0.05], position - 0.4, position, facecolor='white')
    plt.fill_betweenx([0, 0.05], position, position + 0.4, facecolor='white') 
    position = 3
    plt.fill_betweenx([0, 0.05], position, position + 0.4, facecolor='white') 
    plt.xticks(fontproperties=prop, fontsize=14)
    plt.yticks(fontproperties=prop, fontsize=14)
    plt.xlabel('Administrative Scope', fontsize=14, fontproperties=prop)
    plt.ylabel('Correlation Coefficient', fontsize=14, fontproperties=prop)

    plt.grid(True, linestyle='-', linewidth=0.3, color='gray')
    plt.legend(prop=prop)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'save map to {save_path}.')

def hypothesis_testing(before, after):
    _, p_value_normal_before = stats.shapiro(before)
    _, p_value_normal_after = stats.shapiro(after)

    if p_value_normal_before > 0.05 and p_value_normal_after > 0.05:
        _, p_value_t_test = stats.ttest_rel(before, after)
        print("配对样本t检验的P值:", p_value_t_test)
    else:
        _, p_value_wilcoxon = stats.wilcoxon(before, after)
        print("Wilcoxon符号秩检验的P值:", p_value_wilcoxon)


if __name__ == "__main__":

    level = 'county'
    levels = ['county', 'city', 'province']

    which_corr = 'kendalltau'
    corr_types = ['pearson', 'spearmanr', 'kendalltau']
    which_corr_short = {'pearson': 'ps', 'spearmanr': 'sm', 'kendalltau': 'kd'}

    solar_wind_path = 'solar_wind_county_level.geojson'
    solar_wind = gpd.read_file(solar_wind_path)
    print('read solar_wind file completed.')

    output_dir = f'../data_processed/output/fig3'
    os.makedirs(output_dir, exist_ok=True)

    ################################################################################
    # For solar in a region, identify best complementary for following 3 strategies:
    # 1) wind in the same region
    # 2) wind in other regions
    # 3) solar in other regions
    # For wind in a region, identify best complementary for following 3 strategies:
    # 1) solar in the same region
    # 2) solar in other regions
    # 3) wind in other regions
    ################################################################################

    # calculate solar best strategy
    solar_columns = [f'{which_corr}',
               's_w_min_{}'.format(which_corr_short[which_corr]),
               's_s_min_{}'.format(which_corr_short[which_corr])]
    solar_arg_columns = ['',
                   's_w_amin_{}'.format(which_corr_short[which_corr]),
                   's_s_amin_{}'.format(which_corr_short[which_corr])]

    col_solar_min = 's_min_{}'.format(which_corr_short[which_corr])
    col_solar_argmin = 's_amin_{}'.format(which_corr_short[which_corr])

    field = 'power_sum_solar'
    solar_power_sum = np.array(solar_wind[field].tolist())
    solar_idx = np.where(solar_power_sum != 0)[0]
    solar_idx_anti = np.where(solar_power_sum == 0)[0]

    solar_wind[[col_solar_min, col_solar_argmin]] = solar_wind[solar_columns].apply(lambda row: min_value_and_index(row), axis=1)
    solar_wind.loc[solar_idx_anti, col_solar_argmin] = -1


    # calculate wind best strategy
    wind_columns = [f'{which_corr}',
               'w_s_min_{}'.format(which_corr_short[which_corr]),
               'w_w_min_{}'.format(which_corr_short[which_corr])]
    wind_arg_columns = ['',
                   'w_s_amin_{}'.format(which_corr_short[which_corr]),
                   'w_w_amin_{}'.format(which_corr_short[which_corr])]

    col_wind_min = 'w_min_{}'.format(which_corr_short[which_corr])
    col_wind_argmin = 'w_amin_{}'.format(which_corr_short[which_corr])

    field = 'power_sum_wind'
    wind_power_sum = np.array(solar_wind[field].tolist())
    wind_idx = np.where(wind_power_sum != 0)[0]
    wind_idx_anti = np.where(wind_power_sum == 0)[0]

    solar_wind[[col_wind_min, col_wind_argmin]] = solar_wind[wind_columns].apply(lambda row: min_value_and_index(row), axis=1)
    solar_wind.loc[wind_idx_anti, col_wind_argmin] = -1

    # str -> list, unify unit to MW
    solar_wind['power_out_solar'] = solar_wind['power_out_solar'].apply(lambda x: [round(float(val), 4)/1e6 for val in x.strip('[]').split(',')]) # W -> MW
    solar_wind['power_out_wind'] = solar_wind['power_out_wind'].apply(lambda x: [round(float(val), 4) for val in x.strip('[]').split(',')]) # MW


    # solar best strategy matching
    col_solar_match = 's_match'
    solar_wind[col_solar_match] = [[np.nan] for i in range(len(solar_wind))]
    for i, row in solar_wind.iterrows():
        strategy = int(row[col_solar_argmin])
        if strategy == -1:
            continue
        elif strategy == 0:
            solar_wind.at[i, col_solar_match] = np.array(row['power_out_solar']) + np.array(row['power_out_wind'])
        elif strategy == 1: # solar_wind
            arg_column = solar_arg_columns[strategy]
            j = solar_wind[arg_column][i]
            solar_wind.at[i, col_solar_match] = np.array(row['power_out_solar']) + np.array(solar_wind['power_out_wind'][j])
        else: # strategy == 2, solar_solar
            arg_column = solar_arg_columns[strategy]
            j = solar_wind[arg_column][i]
            solar_wind.at[i, col_solar_match] = np.array(row['power_out_solar']) + np.array(solar_wind['power_out_solar'][j])


    # wind best strategy matching
    col_wind_match = 'w_match'
    solar_wind[col_wind_match] = [[np.nan] for i in range(len(solar_wind))]
    for i, row in solar_wind.iterrows():
        strategy = int(row[col_wind_argmin])
        if strategy == -1:
            continue
        elif strategy == 0:
            solar_wind.at[i, col_wind_match] = np.array(row['power_out_wind']) + np.array(row['power_out_solar'])
        elif strategy == 1: # wind_solar
            arg_column = wind_arg_columns[strategy]
            j = solar_wind[arg_column][i]
            solar_wind.at[i, col_wind_match] = np.array(row['power_out_wind']) + np.array(solar_wind['power_out_solar'][j])
        else: # strategy == 2, wind_wind
            arg_column = wind_arg_columns[strategy]
            j = solar_wind[arg_column][i]
            solar_wind.at[i, col_wind_match] = np.array(row['power_out_wind']) + np.array(solar_wind['power_out_wind'][j])


    # before matching
    solar_before = solar_wind['power_out_solar'] # MW
    wind_before = solar_wind['power_out_wind'] # MW
    solar_stat_before = solar_before.apply(calculate_metrics)
    solar_stat_before.columns = ['mean', 'var', 'std', 'mad', 'cv']
    solar_stat_before = solar_stat_before.iloc[solar_idx]
    wind_stat_before = wind_before.apply(calculate_metrics)
    wind_stat_before.columns = ['mean', 'var', 'std', 'mad', 'cv']
    wind_stat_before = wind_stat_before.iloc[wind_idx]


    # after matching
    solar_after = solar_wind['s_match']
    solar_stat_after = solar_after.apply(calculate_metrics)
    solar_stat_after.columns = ['mean', 'var', 'std', 'mad', 'cv']
    solar_stat_after = solar_stat_after.iloc[solar_idx]
    wind_after = solar_wind['w_match']
    wind_stat_after = wind_after.apply(calculate_metrics)
    wind_stat_after.columns = ['mean', 'var', 'std', 'mad', 'cv']
    wind_stat_after = wind_stat_after.iloc[wind_idx]


    # draw CV scatter plot
    stat = 'cv'
    s_before = solar_stat_before[stat]
    s_after = solar_stat_after[stat]
    s_min = min(s_before.min(), s_after.min())
    s_max = max(s_before.max(), s_after.max())
    save_path = os.path.join(output_dir, 'cv_scatter_solar.png')
    scatter_plot_solar(s_before, s_after, save_path)

    w_before = wind_stat_before[stat]
    w_after = wind_stat_after[stat]
    w_min = min(w_before.min(), w_after.min())
    w_max = max(w_before.max(), w_after.max())
    save_path = os.path.join(output_dir, 'cv_scatter_wind.png')
    scatter_plot_wind(w_before, w_after, save_path)


    # draw CV box plot
    data = {'SB': s_before,
        'SA': s_after,
        'WB': w_before,
        'WA': w_after}
    save_path = os.path.join(output_dir, 'cv_boxplot.png')
    boxplot(data, save_path)


    s_before = np.array(s_before)
    s_after = np.array(s_after)
    w_before = np.array(w_before)
    w_after = np.array(w_after)
    print('solar hypothesis testing')
    hypothesis_testing(s_before, s_after)
    print('wind hypothesis testing')
    hypothesis_testing(w_before, w_after)


    # draw CV map
    # colormap = 'PiYG'
    colormap = 'coolwarm'
    alpha = 0.8

    col_solar_cv_before = 's_cv_be'
    solar_wind[col_solar_cv_before] = solar_wind['power_out_solar'].apply(lambda x: np.std(x) / np.mean(x) if len(x) == 8760 else -1)
    legend_label = 'No solar generation'
    cbar_label = 'CV'
    save_path = os.path.join(output_dir, f'cv_map_solar_before_{colormap}_{alpha}.png')
    draw_cv_map(solar_wind, col_solar_cv_before, s_min, s_max, colormap, legend_label, cbar_label, save_path, alpha=alpha)

    col_solar_cv_after = 's_cv_af'
    solar_wind[col_solar_cv_after] = solar_wind['s_match'].apply(lambda x: np.std(x) / np.mean(x) if len(x) == 8760 else -1)
    legend_label = 'No solar generation'
    cbar_label = 'CV'
    save_path = os.path.join(output_dir, f'cv_map_solar_after_{colormap}_{alpha}.png')
    draw_cv_map(solar_wind, col_solar_cv_after, s_min, s_max, colormap, legend_label, cbar_label, save_path, alpha=alpha)

    col_solar_cv_diff = 's_cv_diff'
    solar_wind[col_solar_cv_diff] = solar_wind[col_solar_cv_before] - solar_wind[col_solar_cv_after]
    solar_wind[col_solar_cv_diff][solar_idx_anti] = -1
    s_diff_min = solar_wind[col_solar_cv_diff][solar_idx].min()
    s_diff_max = solar_wind[col_solar_cv_diff][solar_idx].max()
    legend_label = 'No solar generation'
    cbar_label = r'$\Delta$CV'
    save_path = os.path.join(output_dir, f'cv_map_solar_diff_{colormap}.png')
    cbar_ticks = [0.0, 0.2, 0.4, 0.6]
    draw_cv_map(solar_wind, col_solar_cv_diff, s_diff_min, s_diff_max, colormap, legend_label, cbar_label, save_path, alpha=alpha, cbar_ticks=cbar_ticks)

    col_wind_cv_before = 'w_cv_be'
    solar_wind[col_wind_cv_before] = solar_wind['power_out_wind'].apply(lambda x: np.std(x) / np.mean(x) if len(x) == 8760 else -1)
    legend_label = 'No wind generation'
    cbar_label = 'CV'
    save_path = os.path.join(output_dir, f'cv_map_wind_before_{colormap}_{alpha}.png')
    draw_cv_map(solar_wind, col_wind_cv_before, w_min, w_max, colormap, legend_label, cbar_label, save_path, alpha=alpha)

    col_wind_cv_after = 'w_cv_af'
    solar_wind[col_wind_cv_after] = solar_wind['w_match'].apply(lambda x: np.std(x) / np.mean(x) if len(x) == 8760 else -1)
    legend_label = 'No wind generation'
    cbar_label = 'CV'
    save_path = os.path.join(output_dir, f'cv_map_wind_after_{colormap}_{alpha}.png')
    draw_cv_map(solar_wind, col_wind_cv_after, w_min, w_max, colormap, legend_label, cbar_label, save_path, alpha=alpha)

    col_wind_cv_diff = 'w_cv_diff'
    solar_wind[col_wind_cv_diff] = solar_wind[col_wind_cv_before] - solar_wind[col_wind_cv_after]
    solar_wind[col_wind_cv_diff][wind_idx_anti] = -1
    w_diff_min = solar_wind[col_wind_cv_diff][wind_idx].min()
    w_diff_max = solar_wind[col_wind_cv_diff][wind_idx].max()
    legend_label = 'No wind generation'
    cbar_label = r'$\Delta$CV'
    save_path = os.path.join(output_dir, f'cv_map_wind_diff_{colormap}.png')
    cbar_ticks = [-1, 0, 5]
    draw_cv_map(solar_wind, col_wind_cv_diff, w_diff_min, w_diff_max, colormap, legend_label, cbar_label, save_path, alpha=alpha, cbar_ticks=cbar_ticks, twoslopenorm=True)

    # draw violin plot
    which_corr = 'kendalltau'
    solar_wind_path = os.path.join(f'../data_processed/correlation/violinplot/solar_wind_violinplot_{which_corr}.geojson')

    solar_wind = gpd.read_file(solar_wind_path)

    field = 'power_sum_solar'
    solar_power_sum = np.array(solar_wind[field].tolist())
    solar_idx = np.where(solar_power_sum != 0)[0]

    field = 'power_sum_wind'
    wind_power_sum = np.array(solar_wind[field].tolist())
    wind_idx = np.where(wind_power_sum != 0)[0]

    corr_solar_wind_of_same_county = solar_wind[f'{which_corr}'].values
    corr_solar_wind_of_same_county_solar = corr_solar_wind_of_same_county[solar_idx]
    corr_solar_wind_of_same_county_wind = corr_solar_wind_of_same_county[wind_idx]

    min_solar_corr_within_city = solar_wind['s_city'].values[solar_idx]
    min_solar_corr_within_province = solar_wind['s_province'].values[solar_idx]
    min_solar_corr_within_country = solar_wind['s_country'].values[solar_idx]
    min_wind_corr_within_city = solar_wind['w_city'].values[wind_idx]
    min_wind_corr_within_province = solar_wind['w_province'].values[wind_idx]
    min_wind_corr_within_country = solar_wind['w_country'].values[wind_idx]

    solar_len = len(solar_idx)
    wind_len = len(wind_idx)
    corr_data = [corr_solar_wind_of_same_county_solar,
                 min_solar_corr_within_city,
                 min_solar_corr_within_province,
                 min_solar_corr_within_country,
                 corr_solar_wind_of_same_county_wind,
                 min_wind_corr_within_city,
                 min_wind_corr_within_province,
                 min_wind_corr_within_country]

    data = {'Correlation': np.concatenate(corr_data),
            'Level': ['County'] * solar_len + ['City'] * solar_len + ['Province'] * solar_len + ['Country'] * solar_len + 
                     ['County'] * wind_len + ['City'] * wind_len + ['Province'] * wind_len + ['Country'] * wind_len,
            'Type': ['Solar'] * solar_len * 4 + ['Wind'] * wind_len * 4
    }
    save_path = os.path.join(output_dir, f'corr_violinplot_{which_corr}.png')
    draw_violin_plot(data, save_path)
