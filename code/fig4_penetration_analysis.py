# -*- coding: utf-8 -*
import os
import json
import datetime
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import time

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


def storage_calculation(powers, loads, base_load):
    load_met_by_baseload = 0
    load_met_by_power_list = []
    load_met_by_balance_list = []
    load_met_by_storage_list = []
    storage_list = []

    if base_load > 0:
        load_met_by_baseload = base_load
        loads = loads - base_load

    storage = 0
    for power, load in zip(powers, loads):
        excess = power - load
        if excess > 0:
            storage += excess
            load_met_by_power = load
            load_met_by_balance = 0
            load_met_by_storage = 0
        else:
            if storage > -excess:
                storage += excess
                load_met_by_power = power
                load_met_by_balance = 0
                load_met_by_storage = -excess
            else:
                load_met_by_power = power
                load_met_by_balance = -excess - storage
                load_met_by_storage = storage
                storage = 0
        storage_list.append(storage)
        load_met_by_power_list.append(load_met_by_power)
        load_met_by_balance_list.append(load_met_by_balance)
        load_met_by_storage_list.append(load_met_by_storage)

    return storage_list, load_met_by_power_list, load_met_by_balance_list, load_met_by_storage_list

def draw_load_power(power, load, save_path):
    font_path = '/System/Library/Fonts/Helvetica.ttc'
    prop = FontProperties(fname=font_path, size=16)

    time = np.arange(len(power))
    plt.figure(figsize=(10, 4))
    plt.plot(time, load, label='Load', color='#8582BD', alpha=0.3)
    plt.plot(time, power, label='Solar wind generation', color='#EC3E31', alpha=0.3)
    plt.xlabel('Time (hours)', fontsize=16, fontproperties=prop)
    plt.ylabel('Hourly generation (TWh)', fontsize=16, fontproperties=prop)
    plt.xticks(fontproperties=prop)
    plt.yticks(fontproperties=prop)
    leg = plt.legend(loc='upper right', fontsize=16, framealpha=0, prop=prop)
    leg.get_frame().set_linewidth(0)
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return

def draw_load_components(load_met_by_power, load_met_by_balance, load_met_by_storage, base_load, save_path):
    time = np.arange(len(load_met_by_power))
    plt.figure(figsize=(15, 6))
    lw = 0.2
    if base_load > 0:
        plt.fill_between(time, 0, base_load, interpolate=True, color='skyblue', edgecolor='skyblue', alpha=0.3, linewidth=lw, label='Load Met by Baseload Power')
    load_met_by_power = load_met_by_power + base_load
    plt.fill_between(time, base_load, load_met_by_power, interpolate=True, color='green', edgecolor='green', alpha=0.3, linewidth=lw, label='Load Met by Solar Wind Power')
    load_met_by_balance = load_met_by_balance + load_met_by_power
    plt.fill_between(time, load_met_by_power, load_met_by_balance, interpolate=True, color='orange', edgecolor='orange', alpha=0.3, linewidth=lw, label='Load Met by Balance Power')
    if load_met_by_storage.sum() == 0:
        lw = 0
    load_met_by_storage = load_met_by_storage + load_met_by_balance
    plt.fill_between(time, load_met_by_balance, load_met_by_storage, interpolate=True, color='red', edgecolor='red', alpha=0.3, linewidth=lw, label='Load Met by Storage Power')
    plt.xlabel('Time (hours)')
    plt.ylabel('Hourly Power (TW)')
    plt.title('Hourly Composition of Load')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return

def draw_storage(storage, save_path):
    time = np.arange(len(storage))
    plt.figure(figsize=(15, 6))
    plt.fill_between(time, 0, storage, interpolate=True, color='gray', edgecolor='gray', alpha=0.3, linewidth=0.5, label='Storage')
    plt.xlabel('Time (hours)')
    plt.ylabel('Hourly Power (TW)')
    plt.title('Hourly Storage')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return

def autopct_factory():
    value_iter = iter(sizes)
    def autopct(pct):
        val = next(value_iter)
        return '{:.1f}% ({:.1f} TWh)'.format(pct, val)
    return autopct

def draw_pie(labels, sizes, colors, save_path):
    plt.figure(figsize=(10, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct=autopct_factory(), startangle=140, wedgeprops={
        'alpha':0.3, 'edgecolor': 'white', 'linewidth': 2}) # autopct='%1.1f%%'
    plt.axis('equal') 
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return

def draw_bars(balance_country, balance_match, balance_neighbor, balance_province, 
    storage_country, storage_match, storage_neighbor, storage_province, 
    power_country, power_match, power_neighbor, power_province, colors, save_path):
    font_path = '/System/Library/Fonts/Helvetica.ttc'
    prop = FontProperties(fname=font_path, size=10)
    tick_prop = FontProperties(fname=font_path, size=8)
    legend_prop = FontProperties(fname=font_path, size=6)

    flexibilities = ['100%', '90%', '80%', '70%']
    x = np.arange(len(flexibilities))
    width = 0.06

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 6 * width, balance_province, width, label='Flexible generation (A)', color=colors[0], alpha=0.2)
    ax.bar(x - 5 * width, balance_neighbor, width, label='Flexible generation (B)', color=colors[0], alpha=0.4)
    ax.bar(x - 4 * width, balance_match, width, label='Flexible generation (C)', color=colors[0], alpha=0.6)
    ax.bar(x - 3 * width, balance_country, width, label='Flexible generation (D)', color=colors[0], alpha=0.8)

    ax.bar(x - 2 *  width, storage_province, width, label='Storage (A)', color=colors[1], alpha=0.2)
    ax.bar(x - width, storage_neighbor, width, label='Storage (B)', color=colors[1], alpha=0.4)
    ax.bar(x, storage_match, width, label='Storage (C)', color=colors[1], alpha=0.6)
    ax.bar(x + width, storage_country, width, label='Storage (D)', color=colors[1], alpha=0.8)

    ax.bar(x + 2 * width, power_province, width, label='Solar wind generation (A)', color=colors[2], alpha=0.2)
    ax.bar(x + 3 * width, power_neighbor, width, label='Solar wind generation (B)', color=colors[2], alpha=0.4)
    ax.bar(x + 4 * width, power_match, width, label='Solar wind generation (C)', color=colors[2], alpha=0.6)
    ax.bar(x + 5 * width, power_country, width, label='Solar wind generation (D)', color=colors[2], alpha=0.8)

    ax.set_xlabel('System flexibility', fontsize=8, fontproperties=prop)
    ax.set_ylabel('Generation (TWh)', fontsize=8, fontproperties=prop)
    plt.yticks(fontproperties=tick_prop)
    ax.set_xticks(x)
    ax.set_xticklabels(flexibilities)
    for label in ax.get_xticklabels():
        label.set_fontproperties(tick_prop)
    leg = plt.legend(loc='upper right', framealpha=0, prop=legend_prop)
    leg.get_frame().set_linewidth(0)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return


def draw_diff_bars_horizontal(diff_balance1, diff_balance2, diff_balance3,
    diff_storage1, diff_storage2, diff_storage3, diff_power1, diff_power2, diff_power3, colors, save_path):

    font_path = '/System/Library/Fonts/Helvetica.ttc'
    prop = FontProperties(fname=font_path, size=14)

    flexibilities = ['100%', '90%', '80%', '70%']
    y = np.arange(len(flexibilities))
    height = 0.1
    fig, ax = plt.subplots(figsize=(12, 4.5))
    rects1 = ax.barh(y + 4 * height, diff_balance1, height, label='Flexible generation (B - A)', color=colors[0], alpha=0.3)
    rects2 = ax.barh(y + 3 * height, diff_balance2, height, label='Flexible generation (C - A)', color=colors[0], alpha=0.6)
    rects3 = ax.barh(y + 2 * height, diff_balance3, height, label='Flexible generation (D - A)', color=colors[0], alpha=0.9)

    rects4 = ax.barh(y + height, diff_storage1, height, label='Storage (B - A)', color=colors[1], alpha=0.3)
    rects5 = ax.barh(y, diff_storage2, height, label='Storage (C - A)', color=colors[1], alpha=0.6)
    rects6 = ax.barh(y - height, diff_storage3, height, label='Storage (D - A)', color=colors[1], alpha=0.9)

    rects7 = ax.barh(y - 2 * height, diff_power1, height, label='Solar wind generation (B - A)', color=colors[2], alpha=0.3)
    rects8 = ax.barh(y - 3 * height, diff_power2, height, label='Solar wind generation (C - A)', color=colors[2], alpha=0.6)
    rects9 = ax.barh(y - 4 * height, diff_power3, height, label='Solar wind generation (D - A)', color=colors[2], alpha=0.9)

    ax.set_ylabel('System flexibility', fontsize=14, fontproperties=prop)
    ax.set_xlabel('Generation differences (TWh)', fontsize=14, fontproperties=prop)
    # ax.set_title('Energy Differences with Different System Flexibilities')
    plt.xticks(fontproperties=prop)
    ax.set_yticks(y)
    ax.set_yticklabels(flexibilities)
    for label in ax.get_yticklabels():
        label.set_fontproperties(prop)
    # leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=False, ncol=1, fontsize=14, framealpha=0, prop=prop)  # 将图例放置在图表外部
    leg = ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), shadow=False, ncol=1, fontsize=14, framealpha=0, prop=prop)
    leg.get_frame().set_linewidth(0)

    def autolabel(rects):
        for rect in rects:
            width = rect.get_width()
            right_limit = ax.get_xlim()[1]
            left_limit = ax.get_xlim()[0]

            label_position = 3 if width > 0 else -3
            ha = 'left' if width > 0 else 'right'

            ax.annotate('{}'.format(width),
                        xy=(width, rect.get_y() + rect.get_height() / 2),
                        xytext=(label_position, 0),
                        textcoords="offset points",
                        ha=ha, va='center',
                        fontsize=7,
                        fontproperties=prop)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    autolabel(rects6)
    autolabel(rects7)
    autolabel(rects8)
    autolabel(rects9)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return


def draw_storage_bars(province_level, province_neighbor, province_match, country_level, save_path, unit='TWh'):
    font_path = '/System/Library/Fonts/Helvetica.ttc'
    prop = FontProperties(fname=font_path, size=14)

    flexibilities = ['100%', '90%', '80%', '70%']
    x = np.arange(len(flexibilities))
    width = 0.2
    color = '#EC3E31'

    fig, ax = plt.subplots(figsize=(8, 5.5))
    rects1 = ax.bar(x - 2 * width + width / 2, province_level, width, label='Strategy A', color=color, alpha=0.2)
    rects2 = ax.bar(x - width + width / 2, province_neighbor, width, label='Strategy B', color=color, alpha=0.4)
    rects3 = ax.bar(x + width / 2, province_match, width, label='Strategy C', color=color, alpha=0.6)
    rects4 = ax.bar(x + width + width / 2, country_level, width, label='Strategy D', color=color, alpha=0.8)

    ax.set_xlabel('System flexibility', fontsize=14, fontproperties=prop)
    ax.set_ylabel(f'Required storage capacity ({unit})', fontsize=14, fontproperties=prop)
    # ax.set_title('Required Storage Capacity with Different System Flexibilities')
    plt.yticks(fontproperties=prop)
    ax.set_xticks(x)
    ax.set_xticklabels(flexibilities)
    for label in ax.get_xticklabels():
        label.set_fontproperties(prop)
    leg = plt.legend(loc='upper left', framealpha=0, fontsize=14, prop=prop)
    leg.get_frame().set_linewidth(0)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8,
                        fontproperties=prop)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return

def draw_storage_bars_twh(province_level, province_neighbor, province_match, country_level, save_path, unit='TWh'):
    font_path = '/System/Library/Fonts/Helvetica.ttc'
    prop = FontProperties(fname=font_path, size=10)
    tick_prop = FontProperties(fname=font_path, size=8)

    flexibilities = ['100%', '90%', '80%', '70%']
    x = np.arange(len(flexibilities))
    width = 0.2
    color = '#EC3E31'

    fig, ax = plt.subplots(figsize=(8, 4.5))
    rects1 = ax.bar(x - 2 * width + width / 2, province_level, width, label='Strategy A', color=color, alpha=0.2)
    rects2 = ax.bar(x - width + width / 2, province_neighbor, width, label='Strategy B', color=color, alpha=0.4)
    rects3 = ax.bar(x + width / 2, province_match, width, label='Strategy C', color=color, alpha=0.6)
    rects4 = ax.bar(x + width + width / 2, country_level, width, label='Strategy D', color=color, alpha=0.8)

    ax.set_xlabel('System flexibility', fontsize=8, fontproperties=prop)
    ax.set_ylabel(f'Required storage capacity ({unit})', fontsize=8, fontproperties=prop)
    # ax.set_title('Required Storage Capacity with Different System Flexibilities')
    plt.yticks(fontproperties=tick_prop)
    ax.set_xticks(x)
    ax.set_xticklabels(flexibilities)
    for label in ax.get_xticklabels():
        label.set_fontproperties(tick_prop)
    leg = plt.legend(loc='upper left', framealpha=0, fontsize=8, prop=tick_prop)
    leg.get_frame().set_linewidth(0)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=6,
                        fontproperties=prop)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return

def print_save(text, file):
    print(text)
    file.write(text + '\n')

def interpolate_colors(base_colors, total_colors):
    base_colors_rgb = [mcolors.hex2color(c) for c in base_colors]
    num_segments = len(base_colors) - 1
    total_interpolated_colors = total_colors - len(base_colors)
    colors_per_segment = total_interpolated_colors // num_segments
    extra_colors = total_interpolated_colors % num_segments
    expanded_colors = base_colors
    
    for i in range(num_segments):
        start_color = base_colors_rgb[i]
        end_color = base_colors_rgb[i + 1]
        segment_color_count = colors_per_segment + (1 if i < extra_colors else 0)
        for t in np.linspace(0, 1, segment_color_count + 2)[1:-1]:
            interp_color = (1 - t) * np.array(start_color) + t * np.array(end_color)
            expanded_colors.append(mcolors.rgb2hex(interp_color))
    
    if len(expanded_colors) > total_colors:
        expanded_colors = expanded_colors[:total_colors]
    elif len(expanded_colors) < total_colors:
        expanded_colors += [base_colors[-1]] * (total_colors - len(expanded_colors))
    
    return expanded_colors

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

def draw_matches_map_with_load_power_circles(gdf, load_col, power_col, matches, colors, min_val=None, max_val=None, colormap='PiYG', lengend_label='', cbar_label='', save_path='x.png'):
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

    t1 = time.time()
    for idx, row in gdf.iterrows():
        if row['geometry'].geom_type in ['Polygon', 'MultiPolygon'] and row['geometry'].area < 1e-6:
            continue
        # color = sm.to_rgba(row[column]) if row[column] != None else '0.95' # 获取数据对应的颜色
        color = '#D9DEE7'
        ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor=color, edgecolor='black', linewidth=0.1)
        ax_n.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor=color, edgecolor='black', linewidth=0.03)

        # import pdb;pdb.set_trace()
        if row[load_col] != None and row[power_col] != None:
            province = row['省']
            if province == '海南省':
                tmp = gdf[gdf['省'] == province]
                largest_polygon = tmp['geometry'].explode().iloc[[tmp['geometry'].explode().area.argmax()]]
                x, y = largest_polygon.centroid.x, largest_polygon.centroid.y
            else:
                x, y = row['geometry'].centroid.x, row['geometry'].centroid.y
            load_radius = row[load_col] / 3 # np.log(row[load_col] + 1) / 4
            power_radius = row[power_col] / 3 # np.log(row[power_col] + 1) / 4
            # print(f'{province}, load_radius: {load_radius}, power_radius: {power_radius}')
            print(f'{province}, load: {row[load_col]}, power: {row[power_col]}')

            load_c = '#8582BD' #'#AC99D2'
            power_c = '#EC3E31' #'#FBEA2E', '#70CDBE'
            if power_radius < load_radius:
                ax.tissot(rad_km=load_radius, lons=x, lats=y, n_samples=100, facecolor=load_c, alpha=0.9, edgecolor=load_c, linewidth=0.3, zorder=100)
                ax.tissot(rad_km=power_radius, lons=x, lats=y - (load_radius - power_radius) / 111.32, n_samples=100, facecolor=power_c, alpha=0.9, edgecolor=power_c, linewidth=0.3, zorder=100)
                ax_n.tissot(rad_km=load_radius, lons=x, lats=y, n_samples=100, facecolor=load_c, alpha=0.9, edgecolor=load_c, linewidth=0.3, zorder=100)
                ax_n.tissot(rad_km=power_radius, lons=x, lats=y - (load_radius - power_radius) / 111.32, n_samples=100, facecolor=power_c, alpha=0.9, edgecolor=power_c, linewidth=0.3, zorder=100)
            else:
                ax.tissot(rad_km=power_radius, lons=x, lats=y - (load_radius - power_radius) / 111.32, n_samples=100, facecolor=power_c, alpha=0.9, edgecolor=power_c, linewidth=0.3, zorder=100)
                ax.tissot(rad_km=load_radius, lons=x, lats=y, n_samples=100, facecolor=load_c, alpha=0.9, edgecolor=load_c, linewidth=0.3, zorder=100)
                ax_n.tissot(rad_km=power_radius, lons=x, lats=y - (load_radius - power_radius) / 111.32, n_samples=100, facecolor=power_c, alpha=0.9, edgecolor=power_c, linewidth=0.3, zorder=100)
                ax_n.tissot(rad_km=load_radius, lons=x, lats=y, n_samples=100, facecolor=load_c, alpha=0.9, edgecolor=load_c, linewidth=0.3, zorder=100)

    t2 = time.time()
    print(f'complete drawing load and power circles, spend {t2 - t1} seconds.')

    # color = 'black'
    alpha = 1
    lw = 0.6
    # import pdb;pdb.set_trace()
    for i, (province1, province2) in enumerate(matches):
        color = colors[i]

        tmp1 = gdf[gdf['省'] == province1]
        tmp2 = gdf[gdf['省'] == province2]

        if province1 == '海南省':
            largest_polygon = tmp1.geometry.explode().iloc[[tmp1.geometry.explode().area.argmax()]]
            x1, y1 = largest_polygon.centroid.iloc[0].x, largest_polygon.centroid.iloc[0].y
        else:
            x1, y1 = tmp1.geometry.centroid.iloc[0].x, tmp1.geometry.centroid.iloc[0].y

        if province2 == '海南省':
            largest_polygon = tmp2.geometry.explode().iloc[[tmp2.geometry.explode().area.argmax()]]
            x2, y2 = largest_polygon.centroid.iloc[0].x, largest_polygon.centroid.iloc[0].y
        else:
            x2, y2 = tmp2.geometry.centroid.iloc[0].x, tmp2.geometry.centroid.iloc[0].y

        if abs(x2 - x1) > abs(y2 - y1):
            control_x = (x1 + x2) / 2
            control_y = (y1 + y2) / 2 + 0.2 * (x2 - x1)
        else:
            control_x = (x1 + x2) / 2 + 0.2 * (y2 - y1)
            control_y = (y1 + y2) / 2
        path_data = [(x1, y1), (control_x, control_y), (x2, y2)]
        codes = [Path.MOVETO, Path.CURVE3, Path.LINETO]
        path = Path(path_data, codes)
        bezier_patch = PathPatch(path, facecolor='none', edgecolor=color, alpha=alpha, linewidth=lw, zorder=200, transform=ccrs.PlateCarree()._as_mpl_transform(ax))
        ax.add_patch(bezier_patch)

        bezier_patch = PathPatch(path, facecolor='none', edgecolor=color, alpha=0.5, linewidth=0.1, zorder=200, transform=ccrs.PlateCarree()._as_mpl_transform(ax_n))
        ax_n.add_patch(bezier_patch)


    t3 = time.time()
    print(f'complete drawing bezier curves, spend {t3 - t2} seconds.')

    ## legend
    ax.text(128.9, 41, 'Category', fontsize=11, fontproperties=prop, verticalalignment='center', horizontalalignment='left', transform=ccrs.Geodetic())
    ax.tissot(rad_km=100, lons=129.2, lats=38.5, n_samples=100, facecolor=load_c, alpha=0.9, edgecolor=load_c, linewidth=0.3)
    ax.text(129.2 + 2.6, 38.5 - 1, 'Load', fontsize=11, fontproperties=prop, verticalalignment='center', horizontalalignment='left', transform=ccrs.Geodetic())
    ax.tissot(rad_km=100, lons=128.4, lats=36.5, n_samples=100, facecolor=power_c, alpha=0.9, edgecolor=power_c, linewidth=0.3)
    ax.text(128.4 + 2.6, 36.5 - 1, 'Generation', fontsize=11, fontproperties=prop, verticalalignment='center', horizontalalignment='left', transform=ccrs.Geodetic())
    ax.text(126.5, 34.5, 'Quantity (TWh)', fontsize=11, fontproperties=prop, verticalalignment='center', horizontalalignment='left', transform=ccrs.Geodetic())

    base_lon = 126.7
    base_lat = 31.5
    radii = [200, 100, 10, 1] # km
    labels = ['{}'.format(radius * 3) for radius in radii]
    color = 'w'
    longitude_adjustment = [0, -1, -1.65, -2.25]
    lat_offset = 0
    lat_offsets = [0, 3.2, 5.5, 7.5]
    for radius, label, lon_adj, lat_adj in zip(radii, labels, longitude_adjustment, lat_offsets):
        lon = base_lon + lon_adj
        lat = base_lat - lat_adj
        ax.tissot(rad_km=radius, lons=lon, lats=lat, n_samples=100, facecolor=color, alpha=1, edgecolor='black', linewidth=0.3)
        ax.text(lon + 2.6, lat - 1, label, fontsize=11, fontproperties=prop, verticalalignment='center', horizontalalignment='left', transform=ccrs.Geodetic())

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    t4 = time.time()
    print(f'save map to {save_path}. spend {t4 - t3} seconds.')


if __name__ == "__main__":
    __spec__ = None

    ### load data
    load_data_path = '../data_processed/load_data/load_data.json'
    with open(load_data_path, 'r') as f:
        load_data = json.load(f)

    output_dir = '../data_processed/output/fig4'
    os.makedirs(output_dir, exist_ok=True)
    output_dir_load_power = os.path.join(output_dir, 'load_power_matched')
    os.makedirs(output_dir_load_power, exist_ok=True)
    output_dir_load_comp = os.path.join(output_dir, 'load_components')
    os.makedirs(output_dir_load_comp, exist_ok=True)
    output_dir_storage = os.path.join(output_dir, 'storage')
    os.makedirs(output_dir_storage, exist_ok=True)
    output_dir_pie = os.path.join(output_dir, 'pie')
    os.makedirs(output_dir_pie, exist_ok=True)
    output_dir_analysis = os.path.join(output_dir, 'analysis')
    os.makedirs(output_dir_analysis, exist_ok=True)

    log = open(os.path.join(output_dir, 'log.txt'), 'w', encoding='utf-8')

    tolal_load = 0
    # peak_load = 0
    load_country = np.zeros((8760))
    for prov_name_eng, load_prov in load_data.items():
        load_prov = np.array(load_prov) / 1e6 # MW -> TW
        tolal_load += sum(load_prov)
        load_country += load_prov

    ### solar wind power
    solar_wind_path = f'solar_wind_province_level.geojson'
    solar_wind = gpd.read_file(solar_wind_path)

    province_dict = {'北京市':'beijing', '天津市':'tianjin', '河北省':'hebei', '山西省':'shanxi(yi)', '内蒙古自治区':'neimeng', 
                     '辽宁省':'liaoning', '吉林省':'jilin', '黑龙江省':'heilongjiang', '上海市': 'shanghai', '江苏省':'jiangsu',
                     '浙江省':'zhejiang', '安徽省':'anhui', '福建省':'fujian', '江西省':'jiangxi', '山东省':'shandong', '河南省':'henan',
                     '湖北省':'hubei', '湖南省':'hunan', '广东省':'guangdong', '广西壮族自治区':'guangxi', '海南省':'hainan', '重庆市':'chongqing',
                     '四川省':'sichuan', '贵州省':'guizhou', '云南省':'yunnan','西藏自治区':'xizang', '陕西省':'shanxi(san)', '甘肃省':'gansu',
                     '青海省':'qinghai', '宁夏回族自治区':'ningxia', '新疆维吾尔自治区':'xinjiang'}
    neighbors = {
        '北京市': ['天津市', '河北省'],
        '天津市': ['北京市', '河北省'],
        '河北省': ['北京市', '天津市', '山西省', '内蒙古自治区', '辽宁省', '河南省', '山东省'],
        '山西省': ['河北省', '内蒙古自治区', '陕西省', '河南省'],
        '内蒙古自治区': ['河北省', '山西省', '陕西省', '宁夏回族自治区', '甘肃省', '黑龙江省', '吉林省', '辽宁省'],
        '辽宁省': ['内蒙古自治区', '吉林省', '河北省'],
        '吉林省': ['内蒙古自治区', '黑龙江省', '辽宁省'],
        '黑龙江省': ['内蒙古自治区', '吉林省'],
        '上海市': ['江苏省', '浙江省'],
        '江苏省': ['浙江省', '安徽省', '山东省', '上海市'],
        '浙江省': ['江西省', '安徽省', '福建省', '江苏省', '上海市'],
        '安徽省': ['河南省', '湖北省', '江西省', '浙江省', '江苏省', '山东省'],
        '福建省': ['江西省', '广东省', '浙江省'],
        '江西省': ['湖南省', '福建省', '浙江省', '安徽省', '湖北省', '广东省'],
        '山东省': ['河北省', '江苏省', '安徽省', '河南省'],
        '河南省': ['河北省', '山西省', '安徽省', '湖北省', '陕西省', '山东省'],
        '湖北省': ['河南省', '安徽省', '江西省', '湖南省', '重庆市', '陕西省'],
        '湖南省': ['湖北省', '江西省', '广东省', '广西壮族自治区', '贵州省', '重庆市'],
        '广东省': ['福建省', '江西省', '湖南省', '广西壮族自治区'],
        '广西壮族自治区': ['湖南省', '广东省', '贵州省', '云南省'],
        '海南省': [],
        '重庆市': ['湖北省', '湖南省', '贵州省', '四川省', '陕西省'],
        '四川省': ['青海省', '甘肃省', '陕西省', '重庆市', '贵州省', '云南省', '西藏自治区'],
        '贵州省': ['湖南省', '广西壮族自治区', '云南省', '四川省', '重庆市'],
        '云南省': ['西藏自治区', '四川省', '贵州省', '广西壮族自治区'],
        '西藏自治区': ['新疆维吾尔自治区', '青海省', '四川省', '云南省'],
        '陕西省': ['内蒙古自治区', '山西省', '河南省', '湖北省', '重庆市', '四川省', '甘肃省', '宁夏回族自治区'],
        '甘肃省': ['内蒙古自治区', '宁夏回族自治区', '青海省', '四川省', '陕西省', '新疆维吾尔自治区'],
        '青海省': ['新疆维吾尔自治区', '甘肃省', '西藏自治区', '四川省'],
        '宁夏回族自治区': ['内蒙古自治区', '甘肃省', '陕西省'],
        '新疆维吾尔自治区': ['西藏自治区', '青海省', '甘肃省']
    }

    flexibilities = [1, 0.9, 0.8, 0.7]
    base_load_country = [0, 0, 0, 0]
    storage_country = [np.zeros((8760-24)) for i in range(4)]
    load_met_by_power_country = [np.zeros((8760-24)) for i in range(4)]
    load_met_by_balance_country = [np.zeros((8760-24)) for i in range(4)]
    load_met_by_storage_country = [np.zeros((8760-24)) for i in range(4)]
    solar_country = np.zeros((8760))
    wind_country = np.zeros((8760))
    for flexibility in flexibilities:
        save_dir_load_comp = os.path.join(output_dir_load_comp, f'flexibility_{flexibility}')
        os.makedirs(save_dir_load_comp, exist_ok=True)
        save_dir_storage = os.path.join(output_dir_storage, f'flexibility_{flexibility}')
        os.makedirs(save_dir_storage, exist_ok=True)
    
    data = []
    provinces = []
    solar_wind['load'] = None
    solar_wind['s_w_power'] = None
    for prov_name_chi, prov_name_eng in province_dict.items():
        solar_wind_prov = solar_wind[solar_wind['省'] == prov_name_chi]
        solar_prov = solar_wind_prov['power_out_solar'].apply(lambda x: [round(float(val), 4) for val in x.strip('[]').split(',')])
        wind_prov = solar_wind_prov['power_out_wind'].apply(lambda x: [round(float(val), 4) for val in x.strip('[]').split(',')])
        solar_prov = np.array(solar_prov.tolist()[0])
        solar_prov = solar_prov / 1e12
        wind_prov = np.array(wind_prov.tolist()[0])
        wind_prov = wind_prov / 1e6
        solar_country += solar_prov
        wind_country += wind_prov

        if prov_name_eng in load_data:
            load_prov = np.array(load_data[prov_name_eng])
            load_prov = load_prov / 1e6
            # pead_load_prov = load_prov.max()
            load_prov = load_prov[24:]

            solar_wind.loc[solar_wind['省'] == prov_name_chi, 'load'] = load_prov.sum() # TWh

        solar_prov = solar_prov[16:-8]
        wind_prov = wind_prov[16:-8]
        power_prov = solar_prov + wind_prov

        solar_wind.loc[solar_wind['省'] == prov_name_chi, 's_w_power'] = power_prov.sum() # TWh

        item = {}
        item['province'] = prov_name_chi
        item['power'] = power_prov
        item['load'] = load_prov

        # draw load power curve
        save_path = os.path.join(output_dir_load_power, f'load_power_curve_{prov_name_eng}.png')
        draw_load_power(power_prov, load_prov, save_path)

        print_save(f'### {prov_name_chi} ###', log)
        peak_load = load_prov.max()
        min_load = load_prov.min()
        thresh = min_load / peak_load
        print_save(f'load_prov.min()/load_prov.max()={thresh}', log)
        base_load_list = []
        for i, flexibility in enumerate(flexibilities):
            if (1 - flexibility) > thresh:
                base_load = min_load
            else:
                base_load = peak_load * (1 - flexibility)
            base_load_country[i] += base_load
            base_load_list.append(base_load)
            # print(f'peak_load: {peak_load}, base_load: {base_load}')
            load_prov_flex = load_prov - base_load
            assert not np.any(load_prov_flex < 0)
            storage_list, load_met_by_power_list, load_met_by_balance_list, load_met_by_storage_list = storage_calculation(
                power_prov, load_prov, base_load)

            storage_list = np.array(storage_list)
            load_met_by_power_list = np.array(load_met_by_power_list)
            load_met_by_balance_list = np.array(load_met_by_balance_list)
            load_met_by_storage_list = np.array(load_met_by_storage_list)

            storage_country[i] += storage_list
            load_met_by_power_country[i] += load_met_by_power_list
            load_met_by_balance_country[i] += load_met_by_balance_list
            load_met_by_storage_country[i] += load_met_by_storage_list
            total_load_met_by_power = load_met_by_power_list.sum()
            total_load_met_by_balance = load_met_by_balance_list.sum()
            total_load_met_by_storage = load_met_by_storage_list.sum()
            total_load_met_by_baseload = base_load * len(load_met_by_power_list)

            storage_max = storage_list.max()
            storage_non_zero_hours = (storage_list != 0).sum()

            print_save(f'---flexibility: {flexibility}---', log)
            print_save(f'load met by baseload: {total_load_met_by_baseload:.4f}, load met by power: {total_load_met_by_power:.4f}, '
                f'load met by balance: {total_load_met_by_balance:.4f}, load met by storage: {total_load_met_by_storage:.4f}', log)
            print_save(f'storage statistic: max: {storage_max:.4f}, none zero hours: {storage_non_zero_hours}', log)

            # draw load component curve
            save_path = os.path.join(output_dir_load_comp, f'flexibility_{flexibility}/load_component_curve_{prov_name_eng}.png')
            # draw_load_components(load_met_by_power_list, load_met_by_balance_list, load_met_by_storage_list, base_load, save_path)

            # draw storage curve
            if storage_list.sum() > 0:
                save_path = os.path.join(output_dir_storage, f'flexibility_{flexibility}/storage_curve_{prov_name_eng}.png')
                # draw_storage(storage_list, save_path)
        item['baseload'] = base_load_list
        data.append(item)
        provinces.append(prov_name_chi)


    ###
    print_save(f'### Strategy A ###', log)
    total_load_province_level = []
    storage_max_province_level = []
    load_country = load_country[24:]
    for i, (flexibility, load_met_by_power_country_f, load_met_by_balance_country_f, load_met_by_storage_country_f, storage_country_f) in enumerate(zip(
            flexibilities, load_met_by_power_country, load_met_by_balance_country, load_met_by_storage_country, storage_country)):
        total_load_met_by_power_country = load_met_by_power_country_f.sum()
        total_load_met_by_balance_country = load_met_by_balance_country_f.sum()
        total_load_met_by_storage_country = load_met_by_storage_country_f.sum()
        total_load_met_by_baseload_country = base_load_country[i] * len(load_met_by_power_country_f)
        total_load = total_load_met_by_power_country + total_load_met_by_balance_country + total_load_met_by_storage_country + total_load_met_by_baseload_country

        total_load_dict = {}
        total_load_dict['flexibility'] = flexibility
        total_load_dict['total_load_met_by_power'] = total_load_met_by_power_country
        total_load_dict['total_load_met_by_balance'] = total_load_met_by_balance_country
        total_load_dict['total_load_met_by_storage'] = total_load_met_by_storage_country
        total_load_dict['total_load_met_by_baseload'] = total_load_met_by_baseload_country
        total_load_province_level.append(total_load_dict)

        storage_max = storage_country_f.max()
        storage_non_zero_hours = (storage_country_f != 0).sum()
        storage_max_province_level.append(round(storage_max, 2))

        print_save(f'---flexibility: {flexibility}---', log)
        print_save(f'load met by baseload: {total_load_met_by_baseload_country:.4f}, load met by power: {total_load_met_by_power_country:.4f}, '
            f'load met by balance: {total_load_met_by_balance_country:.4f}, load met by storage: {total_load_met_by_storage_country:.4f}, '
            f'total load: {total_load:.4f}', log)
        print_save(f'storage statistic: max: {storage_max:.4f}, none zero hours: {storage_non_zero_hours}', log)

        # draw load component curve
        save_path = os.path.join(output_dir_load_comp, f'flexibility_{flexibility}/load_component_sum_of_provinces.png')
        # draw_load_components(load_met_by_power_country_f, load_met_by_balance_country_f, load_met_by_storage_country_f, 
        #     base_load_country[i], save_path)

        # draw storage curve
        save_path = os.path.join(output_dir_storage, f'flexibility_{flexibility}/storage_curve_sum_of_provinces.png')
        # draw_storage(storage_country_f, save_path)

        # draw pie chart
        sizes = [total_load_met_by_baseload_country, total_load_met_by_power_country, total_load_met_by_balance_country, total_load_met_by_storage_country]
        # sizes = [item / sum(sizes) for item in sizes]
        labels = ['Baseload Power', 'Solar Wind Power', 'Balance Power', 'Storage Power']
        colors = ['skyblue', 'green', 'orange', 'red']
        save_path = os.path.join(output_dir_pie, f'load_component_pie_sum_of_provinces_flexibility_{flexibility}.png')
        # draw_pie(labels=labels, sizes=sizes, colors=colors, save_path=save_path)


    ###
    print_save('### Stragegy C ###', log)
    storage_match = [np.zeros((8760-24)) for i in range(4)]
    load_met_by_power_match = [np.zeros((8760-24)) for i in range(4)]
    load_met_by_balance_match = [np.zeros((8760-24)) for i in range(4)]
    load_met_by_storage_match = [np.zeros((8760-24)) for i in range(4)]
    for fi, flexibility in enumerate(flexibilities):
        # print_save(f'---flexibility: {flexibility}---', log)

        G = nx.Graph()
        num_provinces = len(data)

        for i in range(num_provinces):
            G.add_node(i)

        for i in range(num_provinces):
            for j in range(i + 1, num_provinces):
                power_i = data[i]['power']
                power_j = data[j]['power']
                load_i = data[i]['load']
                load_j = data[j]['load']
                base_load_i = data[i]['baseload'][fi]
                base_load_j = data[j]['baseload'][fi]
                power = power_i + power_j
                load = load_i + load_j
                baseload = base_load_i + base_load_j
                storage_list, load_met_by_power_list, load_met_by_balance_list, load_met_by_storage_list = storage_calculation(
                    power, load, base_load)
                weight = sum(load_met_by_balance_list) + sum(load_met_by_storage_list)
                G.add_edge(i, j, weight=weight)

        mate = nx.algorithms.matching.min_weight_matching(G, weight='weight')

        all_idxs = set(range(num_provinces))
        matches = []
        for m in mate:
            idx1 = m[0]
            idx2 = m[1]
            all_idxs.discard(idx1)
            all_idxs.discard(idx2)
            data1 = data[idx1]
            data2 = data[idx2]
            province1 = data1['province']
            province2 = data2['province']
            matches.append((province1, province2))

            power1, power2 = data1['power'], data2['power']
            load1, load2 = data1['load'], data2['load']
            baseload1, baseload2 = data1['baseload'][fi], data2['baseload'][fi]
            power = power1 + power2
            load = load1 + load2
            baseload = baseload1 + baseload2

            save_path = os.path.join(output_dir_load_power, f'{province1}_{province2}.png')
            draw_load_power(power, load, save_path)

            storage_list, load_met_by_power_list, load_met_by_balance_list, load_met_by_storage_list = storage_calculation(
                power, load, baseload)
            storage_match[fi] += np.array(storage_list)
            load_met_by_power_match[fi] += np.array(load_met_by_power_list)
            load_met_by_balance_match[fi] += np.array(load_met_by_balance_list)
            load_met_by_storage_match[fi] += np.array(load_met_by_storage_list)

        idx = all_idxs.pop()
        data_p = data[idx]
        province = data_p['province']
        power = data_p['power']
        load = data_p['load']
        baseload = data_p['baseload'][fi]
        storage_list, load_met_by_power_list, load_met_by_balance_list, load_met_by_storage_list = storage_calculation(
            power, load, baseload)
        storage_match[fi] += np.array(storage_list)
        load_met_by_power_match[fi] += np.array(load_met_by_power_list)
        load_met_by_balance_match[fi] += np.array(load_met_by_balance_list)
        load_met_by_storage_match[fi] += np.array(load_met_by_storage_list)

    colors = ['#458A74', '#018B38', '#D9A421', '#F5A216', '#57AF37', '#41B9C1', '#008B8B', '#4E5689'] #, '#6A8EC9', '#8A7355', '#848484']
    expanded_colors = interpolate_colors(colors, len(matches))
    print_save(f'{expanded_colors}', log)
    save_path = os.path.join(output_dir, f'arbitrary_matches_with_circle_map.png')

    total_load_province_match = []
    storage_max_province_match = []
    for i, flexibility in enumerate(flexibilities):
        total_load_met_by_power_match = load_met_by_power_match[i].sum()
        total_load_met_by_balance_match = load_met_by_balance_match[i].sum()
        total_load_met_by_storage_match = load_met_by_storage_match[i].sum()
        total_load_met_by_baseload_match = base_load_country[i] * len(load_met_by_power_match[i])
        total_load = total_load_met_by_power_match + total_load_met_by_balance_match + total_load_met_by_storage_match + total_load_met_by_baseload_match

        total_load_dict = {}
        total_load_dict['flexibility'] = flexibility
        total_load_dict['total_load_met_by_power'] = total_load_met_by_power_match
        total_load_dict['total_load_met_by_balance'] = total_load_met_by_balance_match
        total_load_dict['total_load_met_by_storage'] = total_load_met_by_storage_match
        total_load_dict['total_load_met_by_baseload'] = total_load_met_by_baseload_match
        total_load_province_match.append(total_load_dict)

        storage_max = storage_match[i].max()
        storage_non_zero_hours = (storage_match[i] != 0).sum()
        storage_max_province_match.append(round(storage_max, 2))

        print_save(f'---flexibility: {flexibility}---', log)
        print_save(f'load met by baseload: {total_load_met_by_baseload_match:.4f}, load met by power: {total_load_met_by_power_match:.4f}, '
            f'load met by balance: {total_load_met_by_balance_match:.4f}, load met by storage: {total_load_met_by_storage_match:.4f}, '
            f'total load: {total_load:.4f}', log)
        print_save(f'storage statistic: max: {storage_max:.4f}, none zero hours: {storage_non_zero_hours}', log)

        # draw load component curve
        save_path = os.path.join(output_dir_load_comp, f'flexibility_{flexibility}/load_component_sum_of_province_matches.png')
        # draw_load_components(load_met_by_power_match[i], load_met_by_balance_match[i], load_met_by_storage_match[i], 
        #     base_load_country[i], save_path)

        # draw storage curve
        save_path = os.path.join(output_dir_storage, f'flexibility_{flexibility}/storage_curve_sum_of_province_matches.png')
        # draw_storage(storage_match[i], save_path)

        # draw pie chart
        sizes = [total_load_met_by_baseload_match, total_load_met_by_power_match, total_load_met_by_balance_match, total_load_met_by_storage_match]
        # sizes = [item / sum(sizes) for item in sizes]
        labels = ['Baseload Power', 'Solar Wind Power', 'Balance Power', 'Storage Power']
        colors = ['skyblue', 'green', 'orange', 'red']
        save_path = os.path.join(output_dir_pie, f'load_component_pie_sum_of_province_matches_flexibility_{flexibility}.png')
        # draw_pie(labels=labels, sizes=sizes, colors=colors, save_path=save_path)


    ### 
    print_save('### Stragegy B ###', log)
    storage_neighbor = [np.zeros((8760-24)) for i in range(4)]
    load_met_by_power_neighbor = [np.zeros((8760-24)) for i in range(4)]
    load_met_by_balance_neighbor = [np.zeros((8760-24)) for i in range(4)]
    load_met_by_storage_neighbor = [np.zeros((8760-24)) for i in range(4)]
    for fi, flexibility in enumerate(flexibilities):
        print_save(f'---flexibility: {flexibility}---', log)

        G = nx.Graph()
        num_provinces = len(data)

        for i in range(num_provinces):
            G.add_node(i)

        for i in range(num_provinces):
            prov1 = provinces[i]
            for prov2 in neighbors.get(prov1, []):
                if prov2 not in provinces:
                    print('BUG!!!')
                    print(f'{prov1}: {prov2}')
                    continue
                j = provinces.index(prov2)
                power_i = data[i]['power']
                power_j = data[j]['power']
                load_i = data[i]['load']
                load_j = data[j]['load']
                base_load_i = data[i]['baseload'][fi]
                base_load_j = data[j]['baseload'][fi]
                power = power_i + power_j
                load = load_i + load_j
                baseload = base_load_i + base_load_j
                storage_list, load_met_by_power_list, load_met_by_balance_list, load_met_by_storage_list = storage_calculation(
                    power, load, base_load)
                weight = sum(load_met_by_balance_list) + sum(load_met_by_storage_list)
                G.add_edge(i, j, weight=weight)

        mate = nx.algorithms.matching.min_weight_matching(G, weight='weight')

        all_idxs = set(range(num_provinces))
        matches = []
        for m in mate:
            idx1 = m[0]
            idx2 = m[1]
            all_idxs.discard(idx1)
            all_idxs.discard(idx2)
            data1 = data[idx1]
            data2 = data[idx2]
            province1 = data1['province']
            province2 = data2['province']
            matches.append((province1, province2))

            power1, power2 = data1['power'], data2['power']
            load1, load2 = data1['load'], data2['load']
            baseload1, baseload2 = data1['baseload'][fi], data2['baseload'][fi]
            power = power1 + power2
            load = load1 + load2
            baseload = baseload1 + baseload2

            save_path = os.path.join(output_dir_load_power, f'neighbor_{province1}_{province2}.png')
            draw_load_power(power, load, save_path)

            storage_list, load_met_by_power_list, load_met_by_balance_list, load_met_by_storage_list = storage_calculation(
                power, load, baseload)
            storage_neighbor[fi] += np.array(storage_list)
            load_met_by_power_neighbor[fi] += np.array(load_met_by_power_list)
            load_met_by_balance_neighbor[fi] += np.array(load_met_by_balance_list)
            load_met_by_storage_neighbor[fi] += np.array(load_met_by_storage_list)

        idx = all_idxs.pop()
        data_p = data[idx]
        province = data_p['province']
        power = data_p['power']
        load = data_p['load']
        baseload = data_p['baseload'][fi]
        storage_list, load_met_by_power_list, load_met_by_balance_list, load_met_by_storage_list = storage_calculation(
            power, load, baseload)
        storage_neighbor[fi] += np.array(storage_list)
        load_met_by_power_neighbor[fi] += np.array(load_met_by_power_list)
        load_met_by_balance_neighbor[fi] += np.array(load_met_by_balance_list)
        load_met_by_storage_neighbor[fi] += np.array(load_met_by_storage_list)

    save_path = os.path.join(output_dir, f'neighbouring_matches_with_circle_map.png')
    draw_matches_map_with_load_power_circles(solar_wind, 'load', 's_w_power', matches, expanded_colors, save_path=save_path)


    # 配对后汇总
    total_load_province_neighbor = []
    storage_max_province_neighbor = []
    for i, flexibility in enumerate(flexibilities):
        total_load_met_by_power_neighbor = load_met_by_power_neighbor[i].sum()
        total_load_met_by_balance_neighbor = load_met_by_balance_neighbor[i].sum()
        total_load_met_by_storage_neighbor = load_met_by_storage_neighbor[i].sum()
        total_load_met_by_baseload_neighbor = base_load_country[i] * len(load_met_by_power_neighbor[i])
        total_load = total_load_met_by_power_neighbor + total_load_met_by_balance_neighbor + total_load_met_by_storage_neighbor + total_load_met_by_baseload_neighbor

        total_load_dict = {}
        total_load_dict['flexibility'] = flexibility
        total_load_dict['total_load_met_by_power'] = total_load_met_by_power_neighbor
        total_load_dict['total_load_met_by_balance'] = total_load_met_by_balance_neighbor
        total_load_dict['total_load_met_by_storage'] = total_load_met_by_storage_neighbor
        total_load_dict['total_load_met_by_baseload'] = total_load_met_by_baseload_neighbor
        total_load_province_neighbor.append(total_load_dict)

        storage_max = storage_neighbor[i].max()
        storage_non_zero_hours = (storage_neighbor[i] != 0).sum()
        storage_max_province_neighbor.append(round(storage_max, 2))

        print_save(f'---flexibility: {flexibility}---', log)
        print_save(f'load met by baseload: {total_load_met_by_baseload_neighbor:.4f}, load met by power: {total_load_met_by_power_neighbor:.4f}, '
            f'load met by balance: {total_load_met_by_balance_neighbor:.4f}, load met by storage: {total_load_met_by_storage_neighbor:.4f}, '
            f'total load: {total_load:.4f}', log)
        print_save(f'storage statistic: max: {storage_max:.4f}, none zero hours: {storage_non_zero_hours}', log)

        # draw load component curve
        save_path = os.path.join(output_dir_load_comp, f'flexibility_{flexibility}/load_component_sum_of_province_match_neighbor.png')
        # draw_load_components(load_met_by_power_neighbor[i], load_met_by_balance_neighbor[i], load_met_by_storage_neighbor[i], 
        #     base_load_country[i], save_path)

        # draw storage curve
        save_path = os.path.join(output_dir_storage, f'flexibility_{flexibility}/storage_curve_sum_of_province_match_neighbor.png')
        # draw_storage(storage_neighbor[i], save_path)

        # draw pie chart
        sizes = [total_load_met_by_baseload_neighbor, total_load_met_by_power_neighbor, total_load_met_by_balance_neighbor, total_load_met_by_storage_neighbor]
        # sizes = [item / sum(sizes) for item in sizes]
        labels = ['Baseload Power', 'Solar Wind Power', 'Balance Power', 'Storage Power']
        colors = ['skyblue', 'green', 'orange', 'red']
        save_path = os.path.join(output_dir_pie, f'load_component_pie_sum_of_neighbor_province_match_flexibility_{flexibility}.png')
        # draw_pie(labels=labels, sizes=sizes, colors=colors, save_path=save_path)


    ### 
    print_save(f'### Strategy D ###', log)
    solar_wind_country = solar_country + wind_country
    ### 时间对齐
    # load_country = load_country[24:]
    solar_wind_country = solar_wind_country[16:-8]
    peak_load_country = load_country.max()
    thresh = load_country.min()/peak_load_country
    print_save(f'load_country.min()/load_country.max()={thresh}', log)
    # draw country-level power load curve
    save_path = os.path.join(output_dir_load_power, f'load_power_curve_country.png')
    # draw_load_power(solar_wind_country, load_country, save_path)
    total_load_country_level = []
    storage_max_country_level = []
    total_baseload = []
    for i, flexibility in enumerate(flexibilities):
        # base_load_country = peak_load_country * (1 - flexibility)
        # load_country_flex = load_country - base_load_country
        # assert not np.any(load_country_flex < 0)
        storage_list, load_met_by_power_list, load_met_by_balance_list, load_met_by_storage_list = storage_calculation(
                solar_wind_country, load_country, base_load_country[i])
        
        storage_list = np.array(storage_list)
        load_met_by_power_list = np.array(load_met_by_power_list)
        load_met_by_balance_list = np.array(load_met_by_balance_list)
        load_met_by_storage_list = np.array(load_met_by_storage_list)

        total_load_met_by_power = load_met_by_power_list.sum()
        total_load_met_by_balance = load_met_by_balance_list.sum()
        total_load_met_by_storage = load_met_by_storage_list.sum()
        total_load_met_by_baseload = base_load_country[i] * len(load_met_by_power_list)
        total_load = total_load_met_by_power + total_load_met_by_balance + total_load_met_by_storage + total_load_met_by_baseload

        total_load_dict = {}
        total_load_dict['flexibility'] = flexibility
        total_load_dict['total_load_met_by_power'] = total_load_met_by_power
        total_load_dict['total_load_met_by_balance'] = total_load_met_by_balance
        total_load_dict['total_load_met_by_storage'] = total_load_met_by_storage
        total_load_dict['total_load_met_by_baseload'] = total_load_met_by_baseload
        total_load_country_level.append(total_load_dict)
        total_baseload.append(total_load_met_by_baseload)

        storage_max = storage_list.max()
        storage_non_zero_hours = (storage_list != 0).sum()
        storage_max_country_level.append(round(storage_max, 2))

        print_save(f'---flexibility: {flexibility}---', log)
        print_save(f'load met by baseload: {total_load_met_by_baseload:.4f}, load met by power: {total_load_met_by_power:.4f}, '
            f'load met by balance: {total_load_met_by_balance:.4f}, load met by storage: {total_load_met_by_storage:.4f}, '
            f'total load: {total_load:.4f}', log)
        print_save(f'storage statistic: max: {storage_max:.4f}, none zero hours: {storage_non_zero_hours}', log)

        # draw load component curve
        save_path = os.path.join(output_dir_load_comp, f'flexibility_{flexibility}/load_component_curve_country.png')
        # draw_load_components(load_met_by_power_list, load_met_by_balance_list, load_met_by_storage_list, base_load_country[i], save_path)

        # draw storage curve
        save_path = os.path.join(output_dir_storage, f'flexibility_{flexibility}/storage_curve_country.png')
        # draw_storage(storage_list, save_path)

        # draw pie chart
        sizes = [total_load_met_by_baseload, total_load_met_by_power, total_load_met_by_balance, total_load_met_by_storage]
        # sizes = [item / sum(sizes) for item in sizes]
        labels = ['Baseload Power', 'Solar Wind Power', 'Balance Power', 'Storage Power']
        colors = ['skyblue', 'green', 'orange', 'red']
        save_path = os.path.join(output_dir_pie, f'load_component_pie_country_flexibility_{flexibility}.png')
        # draw_pie(labels=labels, sizes=sizes, colors=colors, save_path=save_path)


    ### statistics
    print_save('---Statistics---', log)
    total_solar_wind_power = sum(solar_wind_country)
    print_save(f'total_solar_wind_power: {total_solar_wind_power}', log)
    average_load = load_country.sum() / len(load_country)
    print_save(f'average_load: {average_load} TWh', log)

    ###
    diff_province_neighbor = []
    for province_level, province_neighbor in zip(total_load_province_level, total_load_province_neighbor):
        assert province_level['flexibility'] == province_neighbor['flexibility']
        diff_dict = {}
        diff_dict['flexibility'] = province_level['flexibility']
        for key in ['total_load_met_by_power', 'total_load_met_by_balance', 'total_load_met_by_storage', 'total_load_met_by_baseload']:
            diff_dict[key] = province_neighbor[key] - province_level[key] # TWh
            diff_dict[key] = round(diff_dict[key], 4)
        diff_dict['diff_power_div_total_power'] = round(diff_dict['total_load_met_by_power'] / total_solar_wind_power, 4)
        diff_dict['diff_power_equal_hour_of_avgload'] = round(diff_dict['total_load_met_by_power'] / average_load, 4)
        diff_province_neighbor.append(diff_dict)
    unit = {'unit': 'TWh'}
    diff_province_neighbor.append(unit)
    with open(os.path.join(output_dir_analysis, 'diff_province_neighbor.json'), 'w') as f:
        json.dump(diff_province_neighbor, f)

    ###
    diff_neighbor_match = []
    for province_neighbor, province_match in zip(total_load_province_neighbor, total_load_province_match):
        assert province_neighbor['flexibility'] == province_match['flexibility']
        diff_dict = {}
        diff_dict['flexibility'] = province_neighbor['flexibility']
        for key in ['total_load_met_by_power', 'total_load_met_by_balance', 'total_load_met_by_storage', 'total_load_met_by_baseload']:
            diff_dict[key] = province_match[key] - province_neighbor[key] # TWh
            diff_dict[key] = round(diff_dict[key], 4)
        diff_dict['diff_power_div_total_power'] = round(diff_dict['total_load_met_by_power'] / total_solar_wind_power, 4)
        diff_dict['diff_power_equal_hour_of_avgload'] = round(diff_dict['total_load_met_by_power'] / average_load, 4)
        diff_neighbor_match.append(diff_dict)
    unit = {'unit': 'TWh'}
    diff_neighbor_match.append(unit)
    with open(os.path.join(output_dir_analysis, 'diff_neighbor_match.json'), 'w') as f:
        json.dump(diff_neighbor_match, f)

    ###
    diff_match_country = []
    for province_match, country_level in zip(total_load_province_match, total_load_country_level):
        assert province_match['flexibility'] == country_level['flexibility']
        diff_dict = {}
        diff_dict['flexibility'] = province_match['flexibility']
        for key in ['total_load_met_by_power', 'total_load_met_by_balance', 'total_load_met_by_storage', 'total_load_met_by_baseload']:
            diff_dict[key] = country_level[key] - province_match[key] # TWh
            diff_dict[key] = round(diff_dict[key], 4)
        diff_dict['diff_power_div_total_power'] = round(diff_dict['total_load_met_by_power'] / total_solar_wind_power, 4)
        diff_dict['diff_power_equal_hour_of_avgload'] = round(diff_dict['total_load_met_by_power'] / average_load, 4)
        diff_match_country.append(diff_dict)
    unit = {'unit': 'TWh'}
    diff_match_country.append(unit)
    with open(os.path.join(output_dir_analysis, 'diff_match_country.json'), 'w') as f:
        json.dump(diff_match_country, f)

    ###
    diff_province_match = []
    for province_level, province_match in zip(total_load_province_level, total_load_province_match):
        assert province_level['flexibility'] == province_match['flexibility']
        diff_dict = {}
        diff_dict['flexibility'] = province_level['flexibility']
        for key in ['total_load_met_by_power', 'total_load_met_by_balance', 'total_load_met_by_storage', 'total_load_met_by_baseload']:
            diff_dict[key] = province_match[key] - province_level[key] # TWh
            diff_dict[key] = round(diff_dict[key], 4)
        diff_dict['diff_power_div_total_power'] = round(diff_dict['total_load_met_by_power'] / total_solar_wind_power, 4)
        diff_dict['diff_power_equal_hour_of_avgload'] = round(diff_dict['total_load_met_by_power'] / average_load, 4)
        diff_province_match.append(diff_dict)
    unit = {'unit': 'TWh'}
    diff_province_match.append(unit)
    with open(os.path.join(output_dir_analysis, 'diff_province_match.json'), 'w') as f:
        json.dump(diff_province_match, f)

    ###
    diff_province_country = []
    for province_level, country_level in zip(total_load_province_level, total_load_country_level):
        assert province_level['flexibility'] == country_level['flexibility']
        diff_dict = {}
        diff_dict['flexibility'] = province_level['flexibility']
        for key in ['total_load_met_by_power', 'total_load_met_by_balance', 'total_load_met_by_storage', 'total_load_met_by_baseload']:
            diff_dict[key] = country_level[key] - province_level[key] # TWh
            diff_dict[key] = round(diff_dict[key], 4)
        diff_dict['diff_power_div_total_power'] = round(diff_dict['total_load_met_by_power'] / total_solar_wind_power, 4)
        diff_dict['diff_power_equal_hour_of_avgload'] = round(diff_dict['total_load_met_by_power'] / average_load, 4)
        diff_province_country.append(diff_dict)
    unit = {'unit': 'TWh'}
    diff_province_country.append(unit)
    with open(os.path.join(output_dir_analysis, 'diff_province_country.json'), 'w') as f:
        json.dump(diff_province_country, f)


    ### draw total load bar (country-level and province-level)
    total_power_country = []
    total_power_neighbor = []
    total_power_match = []
    total_power_province = []
    total_balance_country = []
    total_balance_neighbor = []
    total_balance_match = []
    total_balance_province = []
    total_storage_country = []
    total_storage_neighbor = []
    total_storage_match = []
    total_storage_province = []
    for province_level, province_neighbor, province_match, country_level in zip(total_load_province_level, total_load_province_neighbor, total_load_province_match, total_load_country_level):
        assert province_level['flexibility'] == province_neighbor['flexibility'] == province_match['flexibility'] == country_level['flexibility']
        total_power_country.append(round(country_level['total_load_met_by_power']))
        total_power_match.append(round(province_match['total_load_met_by_power']))
        total_power_neighbor.append(round(province_neighbor['total_load_met_by_power']))
        total_power_province.append(round(province_level['total_load_met_by_power']))

        total_balance_country.append(round(country_level['total_load_met_by_balance']))
        total_balance_match.append(round(province_match['total_load_met_by_balance']))
        total_balance_neighbor.append(round(province_neighbor['total_load_met_by_balance']))
        total_balance_province.append(round(province_level['total_load_met_by_balance']))

        total_storage_country.append(round(country_level['total_load_met_by_storage']))
        total_storage_match.append(round(province_match['total_load_met_by_storage']))
        total_storage_neighbor.append(round(province_neighbor['total_load_met_by_storage']))
        total_storage_province.append(round(province_level['total_load_met_by_storage']))
    colors = ['#8582BD', '#EC3E31', '#A6D0E6']
    save_path = os.path.join(output_dir_analysis, 'bar_energy_with_flexibility.png')
    draw_bars(total_balance_country, total_balance_match, total_balance_neighbor, total_balance_province, 
        total_storage_country, total_storage_match, total_storage_neighbor, total_storage_province, 
        total_power_country, total_power_match, total_power_neighbor, total_power_province, colors, save_path)

    # draw ring graph
    ring_data = {}
    for f, bl, p1, p2, p3, p4, b1, b2, b3, b4, s1, s2, s3, s4 in zip(flexibilities, total_baseload, 
            total_power_country, total_power_match, total_power_neighbor, total_power_province,
            total_balance_country, total_balance_match, total_balance_neighbor, total_balance_province,
            total_storage_country, total_storage_match, total_storage_neighbor, total_storage_province):
        A = [bl, p1, b1, s1]
        B = [bl, p2, b2, s2]
        C = [bl, p3, b3, s3]
        D = [bl, p4, b4, s4]
        data = [A, B, C, D]
        data = np.around(np.array(data) / total_load, decimals=2)
        ring_data[f] = data.tolist()
    save_path = os.path.join(output_dir, 'ring_data.json')
    with open(save_path, 'w') as f:
        json.dump(ring_data, f)

    ### draw diff energy bar
    diff_power1 = []
    diff_balance1 = []
    diff_storage1 = []
    diff_power2 = []
    diff_balance2 = []
    diff_storage2 = []
    diff_power3 = []
    diff_balance3 = []
    diff_storage3 = []
    for diff_dict1, diff_dict2, diff_dict3 in zip(diff_province_neighbor, diff_province_match, diff_province_country):
        if 'flexibility' not in diff_dict1 or 'flexibility' not in diff_dict2 or 'flexibility' not in diff_dict3:
            continue
        diff_power1.append((round(diff_dict1['total_load_met_by_power'], 2)))
        diff_balance1.append((round(diff_dict1['total_load_met_by_balance'], 2)))
        diff_storage1.append((round(diff_dict1['total_load_met_by_storage'], 2)))
        diff_power2.append((round(diff_dict2['total_load_met_by_power'], 2)))
        diff_balance2.append((round(diff_dict2['total_load_met_by_balance'], 2)))
        diff_storage2.append((round(diff_dict2['total_load_met_by_storage'], 2)))
        diff_power3.append((round(diff_dict3['total_load_met_by_power'], 2)))
        diff_balance3.append((round(diff_dict3['total_load_met_by_balance'], 2)))
        diff_storage3.append((round(diff_dict3['total_load_met_by_storage'], 2)))
    colors = ['#8582BD', '#EC3E31', '#A6D0E6']
    save_path = os.path.join(output_dir_analysis, 'horizontal_bar_energy_diff_with_flexibility.png')
    draw_diff_bars_horizontal(diff_balance1, diff_balance2, diff_balance3,
        diff_storage1, diff_storage2, diff_storage3, diff_power1, diff_power2, diff_power3, colors, save_path)



    ### draw storage max bar (province-level, province-level matching, and country-level)
    save_path = os.path.join(output_dir_analysis, 'bar_storage_max_with_flexibility.png')
    draw_storage_bars_twh(storage_max_province_level, storage_max_province_neighbor, storage_max_province_match, storage_max_country_level, save_path)
    storage_hour_province_level = [round(st / average_load, 2) for st in storage_max_province_level]
    storage_hour_province_neighbor = [round(st / average_load, 2) for st in storage_max_province_neighbor]
    storage_hour_province_match = [round(st / average_load, 2) for st in storage_max_province_match]
    storage_hour_country_level = [round(st / average_load, 2) for st in storage_max_country_level]
    save_path = os.path.join(output_dir_analysis, 'bar_storage_hour_with_flexibility.png')
    draw_storage_bars(storage_hour_province_level, storage_hour_province_neighbor, storage_hour_province_match, storage_hour_country_level, save_path, unit='hour')

    log.close()

