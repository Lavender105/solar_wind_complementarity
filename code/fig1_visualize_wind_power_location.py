# -*- coding: utf-8 -*
import os
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from copy import copy
import time

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib import rcParams
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import shapely.geometry as sgeom


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

if __name__ == "__main__":
    __spec__ = None

    t1 = time.time()
    county_data_extended_path = '../data/solar_wind_aggregation/solar_wind_qinghai.geojson'
    county_data_extended = gpd.read_file(county_data_extended_path)
    county_data_extended['power_sum_wind'] = county_data_extended['power_sum_wind'].apply(lambda x: x/1e3) # GW
    t2 = time.time()
    print(f'complete reading wind generation data! spend {t2 - t1} seconds.')

    wind_path = '../data/installation_shps/wind/Qinghai.shp'
    wind_data = gpd.read_file(wind_path, encoding='gbk')
    t3 = time.time()
    print(f'complete reading original solar data! spend {t3 - t2} seconds.')

    level2property = {'省': '省', '市': '地名', '县': '地名'}
    level = '县'
    county_boundaries_path = f'../data/China_boundary_shps/2021年{level}矢量.shp'
    county_data = gpd.read_file(county_boundaries_path)
    jiuduanxian_path = f'../data/China_boundary_shps/九段线.shp'
    jiuduanxian = gpd.read_file(jiuduanxian_path)
    t4 = time.time()
    print(f'complete reading county boundary and jiuduanxian! spend {t4 - t3} seconds.')


    output_dir = f'../output/fig1'
    os.makedirs(output_dir, exist_ok=True)

    print('plot power_sum_wind map ...')
    font_path = '/System/Library/Fonts/Helvetica.ttc'
    prop = FontProperties(fname=font_path)
    color_ranges = [(0, 10**0), (10**0, 10**1), (10**1, 10**2), (10**2, 10**3), (10**3, 10**4), (10**4, 10**5)] #, (10**5, 10**6), (10**6, 10**7)]
    colors = [(255, 255, 255), (186, 186, 250), (193, 252, 254), (255, 255, 185), (245, 194, 154), (239, 143, 142)] #, (245, 100, 100), (255, 0, 0)]
    colors = [(r/255, g/255, b/255) for r, g, b in colors]
    cmap = ListedColormap(colors)
    projection = ccrs.LambertConformal(central_longitude=105, central_latitude=35, standard_parallels=(30, 60))
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': projection})
    ax.set_extent([76, 132, 16, 53.5], ccrs.PlateCarree())
    ax_n = fig.add_axes([0.78, 0.23, 0.1, 0.12], projection = projection)
    ax_n.set_extent([104.5, 125, 0, 26])
    t5 = time.time()
    print(f'complete creating fig and ax, ax_n, apend {t5 - t4} seconds')
    fig.canvas.draw()
    t6 = time.time()
    print(f'complete fig.canvas.draw(), spend {t6 - t5} seconds.')
    ax.add_geometries(county_data["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black", linewidth=0.05)
    ax.add_geometries(jiuduanxian["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black", linewidth=0.3)
    t7 = time.time()
    print(f'complete adding coundaries geometry. spend {t7 - t6} seconds.')

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
    t8 = time.time()
    print(f'complete drawing gridlines, spend {t8 - t7} seconds.')

    ax_n.add_geometries(county_data["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black", linewidth=.02)
    ax_n.add_geometries(jiuduanxian["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black", linewidth=.3)
    # fig.canvas.draw()
    xticks = [110, 120]
    yticks = [10, 20]
    gl = ax_n.gridlines(draw_labels=True, x_inline=False, y_inline=False, rotate_labels=0, linewidth=0.1, linestyle='--')
    gl.top_labels = False
    gl.left_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    t9 = time.time()
    print(f'complete drawing nanhai map, apend {t9 - t8} seconds.')


    for i, (start, end) in enumerate(color_ranges):
        subset = county_data_extended[(county_data_extended['power_sum_wind'] >= start) & (county_data_extended['power_sum_wind'] < end)]
        subset = subset[~subset['geometry'].apply(lambda geom: geom.geom_type in ['Polygon', 'MultiPolygon'] and geom.area < 1e-6)]
        if subset.empty:
            continue
        fc = colors[i] if i > 0 else 'None'
        ax.add_geometries(subset["geometry"], crs=ccrs.PlateCarree(), fc=fc, ec="black", linewidth=0.05)
        ax_n.add_geometries(subset["geometry"], crs=ccrs.PlateCarree(), fc=fc, ec="black", linewidth=0.02)
        print('({}, {}): {}'.format(start, end, len(subset['power_sum_wind'])))
    t10 = time.time()
    print(f'complete drawing power_sum_wind, spend {t10 - t9} seconds.')

    print('adding wind geometry...')
    ax.add_geometries(wind_data["geometry"], crs=ccrs.PlateCarree(), fc="darkgreen", ec="darkgreen", linewidth=1)
    # ax.add_geometries(wind_data["geometry"][10000:15000], crs=ccrs.PlateCarree(), fc="darkgreen", ec="darkgreen", linewidth=1)
    t11 = time.time()
    print(f'complete adding! spend {t11 - t10} seconds.')

    sm = ScalarMappable(cmap=cmap)
    ticks = [0, 10**0, 10**1, 10**2, 10**3, 10**4, 10**5] #, 10**6, 10**7]
    sm.set_array(ticks)
    # axins = ax.inset_axes([0.87, 0.32, 0.02, 0.23]) # (x, y, width, height)
    axins = ax.inset_axes([0.85, 0.32, 0.03, 0.33]) # (x, y, width, height)
    cb = plt.colorbar(sm, cax=axins, label='Generation (GWh)')
    cb.set_label('Generation (GWh)', fontsize=14, fontproperties=prop) # fontname=fontname
    cb.set_ticks([tick for tick in np.arange(0, ticks[-1]+1, ticks[-1]/(len(ticks)-1))])
    cb.set_ticklabels([f"$10^{{{int(np.log10(tick))}}}$" if tick != 0 else '0' for tick in ticks], fontsize=14) # fontname=fontname
    for label in cb.ax.get_yticklabels():
        label.set_fontproperties(prop)
        label.set_fontsize(14)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=0) # width=0.5

    sm = ScalarMappable(cmap=ListedColormap(['darkgreen']))
    axins = ax.inset_axes([0.12, 0.06, 0.06, 0.02]) # (x, y, width, height)
    cb = plt.colorbar(sm, cax=axins, label='')
    # cb.set_label('Wind installations', fontsize=8, rotation=0, fontproperties=prop)
    cb.ax.text(1.2, 0.5, 'Wind installations', va='center', ha='left', fontsize=14, fontproperties=prop, transform=cb.ax.transAxes)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=0)
    cb.set_ticklabels([])
    t12 = time.time()
    print(f'complete drawing colorbar, spend {t12 - t11} seconds.')

    save_path = os.path.join(output_dir, f'wind_location_power_map.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    t13 = time.time()
    print(f'complete! save map to {save_path}. spend {t13 - t12} seconds.') # 220s = 3.6min

