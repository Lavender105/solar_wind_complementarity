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

def get_county_name(gdf, idx):
    return '{}{}{}'.format(gdf['省级'].iloc[idx], gdf['地级'].iloc[idx], gdf['地名'].iloc[idx])

def min_value_and_index(row):
    min_value = row.min()
    min_index = row.argmin()
    return pd.Series([min_value, min_index])

def draw_correlation_map_same(gdf, column, save_path):
    font_path = '/System/Library/Fonts/Helvetica.ttc'
    prop = FontProperties(fname=font_path)

    jiuduanxian_path = f'../data/China_boundary_shps/九段线.shp'
    jiuduanxian = gpd.read_file(jiuduanxian_path)

    projection = ccrs.LambertConformal(central_longitude=105, central_latitude=35, standard_parallels=(30, 60))
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': projection})
    ax.set_extent([76, 132, 16, 53.5], ccrs.PlateCarree())  # 设置经纬度范围
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


    cmap = get_cmap('PiYG')
    # norm = Normalize(vmin=gdf[column].min(), vmax=gdf[column].max())
    norm = TwoSlopeNorm(vmin=gdf[column].min(), vcenter=0, vmax=gdf[column].max())
    sm = ScalarMappable(cmap=cmap, norm=norm)
    t1 = time.time()
    for idx, row in gdf.iterrows():
        if row['geometry'].geom_type in ['Polygon', 'MultiPolygon'] and row['geometry'].area < 1e-6:
            continue
        color = sm.to_rgba(row[column]) if row[column] != 0 else '0.95' # 获取数据对应的颜色
        ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor=color, edgecolor='black', linewidth=0.1)
        ax_n.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor=color, edgecolor='black', linewidth=0.03)
    t2 = time.time()
    print(f'complete drawing correlation, spend {t2 - t1} seconds.')

    legend_list = [mpatches.Patch(facecolor='0.95', edgecolor='0.2', linewidth=0.2, label='No solar or wind generation')]

    sm.set_array([])
    vmin = gdf[column].min()
    vmax = gdf[column].max()
    vcenter = 0
    axins = ax.inset_axes([0.83, 0.32, 0.03, 0.33]) # (x, y, width, height)
    cbar = plt.colorbar(sm, cax=axins, label='Correlation coefficient')
    cbar.set_label('Correlation coefficient', fontsize=14, fontproperties=prop)
    cbar.set_ticks([vmin, vcenter, vmax])
    cbar.ax.set_yticklabels([f'{vmin:.2f}', f'{vcenter:.2f}', f'{vmax:.2f}'], fontproperties=prop)
    cbar.ax.tick_params(length=2, width=0.5, labelsize=14)
    cbar.outline.set_linewidth(0.5)
    leg = ax.legend(handles=legend_list, bbox_to_anchor=(0.005, 0.01), loc='lower left', fontsize=14, title='', shadow=False, fancybox=False, framealpha=0) #, prop=prop)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    t3 = time.time()
    print(f'save map to {save_path}. spend {t3 - t2} seconds.')
    

def generate_colors(num_colors):
    hues = np.linspace(0, 1, num_colors, endpoint=False)
    colors = [plt.cm.hsv(hue) for hue in hues]
    return colors

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

def draw_correlation_bezier_map_diff(gdf, column_min, column_argmin, colors, legend_label, save_path, kwargs):
    lw = kwargs['linewidth']
    alpha = kwargs['alpha']

    font_path = '/System/Library/Fonts/Helvetica.ttc'
    prop = FontProperties(fname=font_path)

    jiuduanxian_path = f'../data/China_boundary_shps/九段线.shp'
    jiuduanxian = gpd.read_file(jiuduanxian_path)

    projection = ccrs.LambertConformal(central_longitude=105, central_latitude=35, standard_parallels=(30, 60))
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': projection})
    ax.set_extent([76, 132, 16, 53.5], ccrs.PlateCarree())  # 设置经纬度范围
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


    cmap = get_cmap('PiYG')
    # norm = Normalize(vmin=gdf[column_min].min(), vmax=gdf[column_min].max())
    norm = Normalize(vmin=-0.26, vmax=gdf[column_min].max())
    sm = ScalarMappable(cmap=cmap, norm=norm)
    t1 = time.time()
    for idx, row in gdf.iterrows():
        if row['geometry'].geom_type in ['Polygon', 'MultiPolygon'] and row['geometry'].area < 1e-6:
            continue
        color = sm.to_rgba(row[column_min]) if row[column_min] != 0 else '0.95' # 获取数据对应的颜色
        ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor=color, edgecolor='black', linewidth=0.1)
        ax_n.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor=color, edgecolor='black', linewidth=0.03)
    t2 = time.time()
    print(f'complete drawing correlation, spend {t2 - t1} seconds.')

    legend_list = [mpatches.Patch(facecolor='0.95', edgecolor='0.2', linewidth=0.2, label=legend_label)]

    color_dict = {}
    for des, color in zip(destinations, colors):
        color_dict[des] = color

    for i in range(len(gdf)):
        j = gdf[column_argmin][i]
        if j == -1:
            continue
        color = color_dict[j]
        x1, y1 = gdf.geometry[i].centroid.x, gdf.geometry[i].centroid.y
        x2, y2 = gdf.geometry[j].centroid.x, gdf.geometry[j].centroid.y
        if abs(x2 - x1) > abs(y2 - y1):
            control_x = (x1 + x2) / 2
            control_y = (y1 + y2) / 2 + 0.2 * (x2 - x1)
        else:
            control_x = (x1 + x2) / 2 + 0.2 * (y2 - y1)
            control_y = (y1 + y2) / 2
        path_data = [(x1, y1), (control_x, control_y), (x2, y2)]
        codes = [Path.MOVETO, Path.CURVE3, Path.LINETO]
        path = Path(path_data, codes)
        bezier_patch = PathPatch(path, facecolor='none', edgecolor=color, alpha=alpha, linewidth=lw, zorder=3, transform=ccrs.PlateCarree()._as_mpl_transform(ax))
        ax.add_patch(bezier_patch)
        circle = Circle((x2, y2), radius=0.2, facecolor=(1,1,1,0.2), edgecolor=color, linewidth=0.5, zorder=4, transform=ccrs.PlateCarree()._as_mpl_transform(ax))
        ax.add_patch(circle)

        bezier_patch = PathPatch(path, facecolor='none', edgecolor=color, alpha=0.5, linewidth=0.1, zorder=3, transform=ccrs.PlateCarree()._as_mpl_transform(ax_n))
        ax_n.add_patch(bezier_patch)
        circle = Circle((x2, y2), radius=0.2, facecolor=(1,1,1,0.2), edgecolor=color, linewidth=0.5, zorder=4, transform=ccrs.PlateCarree()._as_mpl_transform(ax_n))
        ax_n.add_patch(circle)


    t3 = time.time()
    print(f'complete drawing bezier curves, spend {t3 - t2} seconds.')

    sm.set_array([])
    vmin = gdf[column].min()
    vmax = gdf[column].max()
    vcenter = 0
    axins = ax.inset_axes([0.83, 0.32, 0.03, 0.33]) # (x, y, width, height)
    cbar = plt.colorbar(sm, cax=axins, label='Correlation coefficient')
    cbar.set_label('Correlation coefficient', fontsize=14, fontproperties=prop)
    cbar.set_ticks([vmin, vcenter, vmax])
    cbar.ax.set_yticklabels([f'{vmin:.2f}', f'{vcenter:.2f}', f'{vmax:.2f}'], fontproperties=prop)
    cbar.ax.tick_params(length=2, width=0.5, labelsize=14)
    cbar.outline.set_linewidth(0.5)
    leg = ax.legend(handles=legend_list, bbox_to_anchor=(0.005, 0.01), loc='lower left', fontsize=14, title='', shadow=False, fancybox=False, framealpha=0) #, prop=prop)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    t4 = time.time()
    print(f'save map to {save_path}. spend {t4 - t3} seconds.')

def draw_correlation_map_diff(gdf, column_min, column_argmin, colors, legend_label, save_path, kwargs):
    lw = kwargs['linewidth']
    alpha = kwargs['alpha']

    font_path = '/System/Library/Fonts/Helvetica.ttc'
    prop = FontProperties(fname=font_path)

    jiuduanxian_path = f'../data/China_boundary_shps/九段线.shp'
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


    cmap = get_cmap('PiYG')
    # norm = Normalize(vmin=gdf[column_min].min(), vmax=gdf[column_min].max())
    norm = Normalize(vmin=-0.26, vmax=gdf[column_min].max())
    sm = ScalarMappable(cmap=cmap, norm=norm)
    t1 = time.time()
    for idx, row in gdf.iterrows():
        if row['geometry'].geom_type in ['Polygon', 'MultiPolygon'] and row['geometry'].area < 1e-6:
            continue
        color = sm.to_rgba(row[column_min]) if row[column_min] != 0 else '0.95' # 获取数据对应的颜色
        ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor=color, edgecolor='black', linewidth=0.1)
        ax_n.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor=color, edgecolor='black', linewidth=0.03)
    t2 = time.time()
    print(f'complete drawing correlation, spend {t2 - t1} seconds.')

    legend_list = [mpatches.Patch(facecolor='0.95', edgecolor='0.2', linewidth=0.2, label=legend_label)]

    sm.set_array([])
    vmin = gdf[column_min].min()
    vmax = gdf[column_min].max()
    vcenter = 0
    axins = ax.inset_axes([0.83, 0.32, 0.03, 0.33]) # (x, y, width, height)
    cbar = plt.colorbar(sm, cax=axins, label='Correlation coefficient')
    cbar.set_label('Correlation coefficient', fontsize=14, fontproperties=prop)
    cbar.set_ticks([vmin, vcenter, vmax])
    cbar.ax.set_yticklabels([f'{vmin:.2f}', f'{vcenter:.2f}', f'{vmax:.2f}'], fontproperties=prop)
    cbar.ax.tick_params(length=2, width=0.5, labelsize=14)
    cbar.outline.set_linewidth(0.5)
    leg = ax.legend(handles=legend_list, bbox_to_anchor=(0.005, 0.01), loc='lower left', fontsize=14, title='', shadow=False, fancybox=False, framealpha=0) #, prop=prop)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'save map to {save_path}.')

def draw_strategy_map(gdf, strategies, column_argmin, column_min, cbar_label, title, legend_labels, save_path):
    font_path = '/System/Library/Fonts/Helvetica.ttc'
    prop = FontProperties(fname=font_path)
    bold_prop = FontProperties(fname=font_path, weight='bold')

    jiuduanxian_path = f'../data/China_boundary_shps/九段线.shp'
    jiuduanxian = gpd.read_file(jiuduanxian_path)

    projection = ccrs.LambertConformal(central_longitude=105, central_latitude=35, standard_parallels=(30, 60))
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': projection})
    ax.set_extent([76, 132, 16, 53.5], ccrs.PlateCarree())  # 设置经纬度范围
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


    cmap = get_cmap('PiYG')
    # norm = Normalize(vmin=gdf[column_min].min(), vmax=gdf[column_min].max())
    norm = Normalize(vmin=-0.26, vmax=gdf[column_min].max())
    sm = ScalarMappable(cmap=cmap, norm=norm)

    t1 = time.time()
    plt.rcParams['hatch.linewidth'] = 0.4
    hatch_styles = ['......', '------', '||||||', None]
    hatch_styles_n = ['........', '--------', '||||||||', None]
    legend_list = []
    for i in strategies:
        tmp = gdf[gdf[column_argmin] == i]
        if len(tmp) != 0:
            # color = 'w' if i >= 0 else 'w'
            for _, row in tmp.iterrows():
                if row['geometry'].geom_type in ['Polygon', 'MultiPolygon'] and row['geometry'].area < 1e-6:
                    continue
                color = sm.to_rgba(row[column_min]) if row[column_min] != 0 else '0.95' # 获取数据对应的颜色
                paths = geos_to_path(row['geometry'])
                for path in paths:
                    patch = PathPatch(path, transform=ccrs.PlateCarree()._as_mpl_transform(ax),
                                      facecolor=color, edgecolor='black', linewidth=0.1,
                                      hatch=hatch_styles[i])
                    ax.add_patch(patch)
                    patch = PathPatch(path, transform=ccrs.PlateCarree()._as_mpl_transform(ax_n),
                                      facecolor=color, edgecolor='black', linewidth=0.03,
                                      hatch=hatch_styles_n[i])
                    ax_n.add_patch(patch)
        if i >= 0:
            legend_list.append(
                mpatches.Patch(facecolor='w', edgecolor='0.2', linewidth=0.2, label=legend_labels[i], hatch=hatch_styles[i])
            )
    legend_list.append(
            mpatches.Patch(facecolor='0.95', edgecolor='0.2', linewidth=0.2, label=legend_labels[-1])
        )
    t2 = time.time()
    print(f'complete drawing strategy hatch, spend {t2 - t1} seconds.')

    sm = ScalarMappable(cmap=cmap)
    sm.set_array([])
    vmin = -0.26 # gdf[column_min].min()
    vmax = gdf[column_min].max()
    vcenter = round((vmin + vmax) / 2, 2)
    sm.set_clim(vmin=vmin, vmax=vmax)
    axins = ax.inset_axes([0.83, 0.32, 0.03, 0.33]) # (x, y, width, height)
    cbar = plt.colorbar(sm, cax=axins, label='Correlation coefficient')
    cbar.set_label('Correlation coefficient', fontsize=14, fontproperties=prop)
    cbar.set_ticks([vmin, vcenter, vmax])
    cbar.ax.set_yticklabels([f'{vmin:.2f}', f'{vcenter:.2f}', f'{vmax:.2f}'], fontproperties=prop)
    cbar.ax.tick_params(length=2, width=0.5, labelsize=14)
    cbar.outline.set_linewidth(0.5)
    leg = ax.legend(handles=legend_list, bbox_to_anchor=(0.005, 0.0), loc='lower left', fontsize=10, title='Strategy', shadow=False, fancybox=False, framealpha=0)#, prop=prop)
    # leg.set_title('Strategy', prop=bold_prop)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    t3 = time.time()
    print(f'save map to {save_path}. spend {t3 - t2} seconds.')



if __name__ == "__main__":

    level = 'county'
    levels = ['county', 'city', 'province']

    which_corr = 'kendalltau'
    corr_types = ['pearson', 'spearmanr', 'kendalltau']
    which_corr_short = {'pearson': 'ps', 'spearmanr': 'sm', 'kendalltau': 'kd'}

    solar_wind_path = '../data/solar_wind_aggregation/solar_wind_qinghai.geojson'
    solar_wind = gpd.read_file(solar_wind_path)
    print('read solar_wind file completed.')

    output_dir = f'../output/fig2'
    os.makedirs(output_dir, exist_ok=True)

    ################################################################################
    # For solar in a region, identify best complementary for following 3 strategies:
    # 1) wind in the same region
    # 2) wind in other regions
    # 3) solar in other regions
    ################################################################################

    # # fig2-a
    col_solar_same = f'{which_corr}'
    save_path = os.path.join(output_dir, f'solar_wind_corr_same_region_{which_corr}.png')
    draw_correlation_map_same(solar_wind, col_solar_same, save_path)


    # # fig2-b
    col_solar_wind_diff = 's_w_min_{}'.format(which_corr_short[which_corr])
    col_solar_wind_diff_arg = 's_w_amin_{}'.format(which_corr_short[which_corr])
    destinations = [i for i in np.unique(solar_wind[col_solar_wind_diff_arg]) if i != -1]
    colors = ['#F8B072', '#8582BD', '#4F99C9', '#A8D3A0', '#A6D0E6']
    legend_label = 'No solar generation or CC > 0'
    kwargs =  {'alpha': 0.5, 'linewidth': 0.2}
    save_path = os.path.join(output_dir, f'solar_wind_diff_region_corr_bezier_{which_corr}.png')
    # draw_correlation_bezier_map_diff(solar_wind, col_solar_wind_diff, col_solar_wind_diff_arg, colors, legend_label, save_path, kwargs)
    save_path = os.path.join(output_dir, f'solar_wind_diff_region_corr_{which_corr}.png')
    draw_correlation_map_diff(solar_wind, col_solar_wind_diff, col_solar_wind_diff_arg, colors, legend_label, save_path, kwargs)

    # fig2-e strategy map
    columns = [f'{which_corr}',
               's_w_min_{}'.format(which_corr_short[which_corr]),
               's_s_min_{}'.format(which_corr_short[which_corr])]

    col_solar_min = 's_min_{}'.format(which_corr_short[which_corr])
    col_solar_argmin = 's_amin_{}'.format(which_corr_short[which_corr])

    field = 'power_sum_solar'
    solar_power_sum = np.array(solar_wind[field].tolist())
    solar_idx = np.where(solar_power_sum != 0)[0]
    solar_idx_anti = np.where(solar_power_sum == 0)[0]

    solar_wind[[col_solar_min, col_solar_argmin]] = solar_wind[columns].apply(lambda row: min_value_and_index(row), axis=1)
    solar_wind.loc[solar_idx_anti, col_solar_argmin] = -1

    strategies = [-1, 0, 1, 2]
    legend_labels = ['Wind in the same regions', 'Wind in different regions',
        'Solar in different regions', 'No solar generation']
    save_path = os.path.join(output_dir, f'solar_strategy_map_{which_corr}.png')
    draw_strategy_map(gdf=solar_wind,
                      strategies=strategies,
                      column_argmin=col_solar_argmin,
                      column_min=col_solar_min,
                      cbar_label='Strategy',
                      title='Correlation coefficient under the optimal complementary strategy for solar',
                      legend_labels=legend_labels,
                      save_path=save_path)
    print(f'draw solar strategy map completed.')

    ################################################################################
    # For wind in a region, identify best complementary for following 3 strategies:
    # 1) solar in the same region
    # 2) solar in other regions
    # 3) wind in other regions
    ################################################################################

    # fig2-c
    col_wind_solar_diff = 'w_s_min_{}'.format(which_corr_short[which_corr])
    col_wind_solar_diff_arg = 'w_s_amin_{}'.format(which_corr_short[which_corr])
    destinations = [i for i in np.unique(solar_wind[col_wind_solar_diff_arg]) if i != -1]
    colors = ['#FBEA2E', '#F8B072', '#8582BD', '#4F99C9', '#A8D3A0', '#A6D0E6', '#EC3E31']
    expanded_colors = interpolate_colors(colors, len(destinations))
    legend_label = 'No wind generation or CC > 0'
    save_path = os.path.join(output_dir, f'wind_solar_diff_region_corr_bezier_{which_corr}.png')
    kwargs =  {'alpha': 0.6, 'linewidth': 0.3}
    # draw_correlation_bezier_map_diff(solar_wind, col_wind_solar_diff, col_wind_solar_diff_arg, expanded_colors, legend_label, save_path, kwargs)
    save_path = os.path.join(output_dir, f'wind_solar_diff_region_corr_{which_corr}.png')
    draw_correlation_map_diff(solar_wind, col_wind_solar_diff, col_wind_solar_diff_arg, expanded_colors, legend_label, save_path, kwargs)

    # fig2-d
    col_wind_wind_diff = 'w_w_min_{}'.format(which_corr_short[which_corr])
    col_wind_wind_diff_arg = 'w_w_amin_{}'.format(which_corr_short[which_corr])
    destinations = [i for i in np.unique(solar_wind[col_wind_wind_diff_arg]) if i != -1]
    colors = ['#FBEA2E', '#F8B072', '#8582BD', '#4F99C9', '#A8D3A0', '#A6D0E6', '#EC3E31']
    expanded_colors = interpolate_colors(colors, len(destinations))
    legend_label = 'No wind generation or CC > 0'
    save_path = os.path.join(output_dir, f'wind_wind_diff_region_corr_bezier_{which_corr}.png')
    kwargs =  {'alpha': 0.6, 'linewidth': 0.25}
    # draw_correlation_bezier_map_diff(solar_wind, col_wind_wind_diff, col_wind_wind_diff_arg, expanded_colors, legend_label,  save_path, kwargs)
    save_path = os.path.join(output_dir, f'wind_wind_diff_region_corr_{which_corr}.png')
    draw_correlation_map_diff(solar_wind, col_wind_wind_diff, col_wind_wind_diff_arg, expanded_colors, legend_label,  save_path, kwargs)

    # fig2-f
    columns = [f'{which_corr}',
               'w_s_min_{}'.format(which_corr_short[which_corr]),
               'w_w_min_{}'.format(which_corr_short[which_corr])]

    col_wind_min = 'w_min_{}'.format(which_corr_short[which_corr])
    col_wind_argmin = 'w_amin_{}'.format(which_corr_short[which_corr])

    field = 'power_sum_wind'
    wind_power_sum = np.array(solar_wind[field].tolist())
    wind_idx = np.where(wind_power_sum != 0)[0]
    wind_idx_anti = np.where(wind_power_sum == 0)[0]

    solar_wind[[col_wind_min, col_wind_argmin]] = solar_wind[columns].apply(lambda row: min_value_and_index(row), axis=1)
    solar_wind.loc[wind_idx_anti, col_wind_argmin] = -1

    ### draw map
    strategies = [-1, 0, 1, 2]
    legend_labels = ['Solar in the same regions', 'Solar in different regions',
        'Wind in different regions', 'No wind generation']
    save_path = os.path.join(output_dir, f'wind_strategy_map_{which_corr}.png')
    draw_strategy_map(gdf=solar_wind,
                      strategies=strategies,
                      column_argmin=col_wind_argmin,
                      column_min=col_wind_min,
                      cbar_label='Strategy',
                      title='Correlation coefficient under the optimal complementary strategy for wind',
                      legend_labels=legend_labels,
                      save_path=save_path)
    print(f'draw wind strategy map completed.')
