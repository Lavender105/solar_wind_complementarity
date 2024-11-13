# -*- coding: utf-8 -*
import os
import json
import datetime
import pandas as pd
import numpy as np
import geopandas as gpd
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


def balance_calculation_with_storage(powers, loads, base_load):
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

def balance_calculation_without_storage(powers, loads, base_load):
    load_met_by_baseload = 0
    load_met_by_power_list = []
    load_met_by_balance_list = []

    if base_load > 0:
        load_met_by_baseload = base_load
        loads = loads - base_load

    for power, load in zip(powers, loads):
        excess = power - load
        if excess > 0:
            load_met_by_power = load
            load_met_by_balance = 0
        else:
            load_met_by_power = power
            load_met_by_balance = - excess

        load_met_by_power_list.append(load_met_by_power)
        load_met_by_balance_list.append(load_met_by_balance)

    return load_met_by_power_list, load_met_by_balance_list

def draw_load_components(load_met_by_power, load_met_by_balance, load_met_by_storage=None, base_load=0, save_path='./'):
    font_path = '/System/Library/Fonts/Helvetica.ttc'
    prop = FontProperties(fname=font_path, size=20)

    time = np.arange(len(load_met_by_power))
    plt.figure(figsize=(15, 6))
    lw = 0.2
    if base_load > 0:
        plt.fill_between(time, 0, base_load, interpolate=True, color='skyblue', edgecolor='skyblue', alpha=0.3, linewidth=lw, label='Load met by baseload generation')
    load_met_by_power = load_met_by_power + base_load
    plt.fill_between(time, base_load, load_met_by_power, interpolate=True, color='green', edgecolor='green', alpha=0.3, linewidth=lw, label='Load met by solar wind generation')
    load_met_by_balance = load_met_by_balance + load_met_by_power
    plt.fill_between(time, load_met_by_power, load_met_by_balance, interpolate=True, color='orange', edgecolor='orange', alpha=0.3, linewidth=lw, label='Load met by flexible generation')
    if load_met_by_storage is not None:
        if load_met_by_storage.sum() == 0:
            lw = 0
        load_met_by_storage = load_met_by_storage + load_met_by_balance
        plt.fill_between(time, load_met_by_balance, load_met_by_storage, interpolate=True, color='red', edgecolor='red', alpha=0.3, linewidth=lw, label='Load met by storage')
    plt.xlabel('Time (hours)', fontsize=20, fontproperties=prop)
    plt.ylabel('Hourly power (TW)', fontsize=20, fontproperties=prop)
    # plt.title('Hourly Composition of Load')
    plt.xticks(fontproperties=prop)
    plt.yticks(fontproperties=prop)
    leg = plt.legend(fontsize=20, framealpha=0, prop=prop)
    leg.get_frame().set_linewidth(0)
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'save map to {save_path}.')


def draw_bars(total_balance_w_st, total_balance_wo_st, save_path):
    font_path = '/System/Library/Fonts/Helvetica.ttc'
    prop = FontProperties(fname=font_path, size=10)

    flexibilities = ['100%', '90%', '80%', '70%']
    x = np.arange(len(flexibilities))
    width = 0.3

    fig, ax = plt.subplots(figsize=(5, 4))
    rects1 = ax.bar(x - width + width/2, total_balance_wo_st, width, label='Without storage', color='orange', alpha=0.25, edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x + width/2 , total_balance_w_st, width, label='With storage', color='orange', alpha=0.5, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('System flexibility', fontsize=10, fontproperties=prop)
    ax.set_ylabel(f'Flexible generation (TWh)', fontsize=10, fontproperties=prop)
    ax.set_xticks(x)
    ax.set_xticklabels(flexibilities, fontproperties=prop)
    for label in ax.get_yticklabels():
        label.set_fontproperties(prop)
    leg = plt.legend(loc='upper right', framealpha=0, prop=prop)
    leg.get_frame().set_linewidth(0)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 2),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8,
                        fontproperties=prop)

    autolabel(rects1)
    autolabel(rects2)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def draw_diff_bars_twh(diff_balance, save_path, unit='TWh'):
    font_path = '/System/Library/Fonts/Helvetica.ttc'
    prop = FontProperties(fname=font_path, size=10)

    flexibilities = ['100%', '90%', '80%', '70%']
    x = np.arange(len(flexibilities))
    width = 0.4

    fig, ax = plt.subplots(figsize=(5, 4))
    rects1 = ax.bar(x, diff_balance, width, label='', color='orange', alpha=0.4)

    ax.set_xlabel('System Flexibility', fontsize=10, fontproperties=prop)
    ax.set_ylabel(f'Increased flexible generation ({unit})', fontsize=10, fontproperties=prop)
    ax.set_xticks(x)
    ax.set_xticklabels(flexibilities, fontproperties=prop)
    for label in ax.get_yticklabels():
        label.set_fontproperties(prop)
    leg = plt.legend(loc='upper left', framealpha=0, prop=prop)
    leg.get_frame().set_linewidth(0)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8,
                        fontproperties=prop)

    autolabel(rects1)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return


def draw_diff_bars(diff_balance, save_path, unit='TWh'):
    font_path = '/System/Library/Fonts/Helvetica.ttc'
    prop = FontProperties(fname=font_path, size=14)

    flexibilities = ['100%', '90%', '80%', '70%']
    x = np.arange(len(flexibilities))  # 横坐标位置
    width = 0.4

    fig, ax = plt.subplots(figsize=(4.8, 4))
    rects1 = ax.bar(x, diff_balance, width, label='', color='orange', alpha=0.4)

    ax.set_xlabel('System Flexibility', fontsize=14, fontproperties=prop)
    ax.set_ylabel(f'Increased flexible generation ({unit})', fontsize=14, fontproperties=prop)
    ax.set_xticks(x)
    ax.set_xticklabels(flexibilities, fontproperties=prop)
    for label in ax.get_yticklabels():
        label.set_fontproperties(prop)
    leg = plt.legend(loc='upper left', framealpha=0, prop=prop)
    leg.get_frame().set_linewidth(0)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10,
                        fontproperties=prop)

    autolabel(rects1)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return

def print_save(text, file):
    print(text)
    file.write(text + '\n')

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

def draw_map(gdf, column, min_val=None, max_val=None, colormap='PiYG', lengend_label='', cbar_label='', save_path='x.png'):
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
    xticks = [110, 120]
    yticks = [10, 20]
    gl = ax_n.gridlines(draw_labels=True, x_inline=False, y_inline=False, rotate_labels=0, linewidth=0.1, linestyle='--')
    gl.top_labels = False
    gl.left_labels = False
    gl.right_labels = False
    gl.bottom_labels = False

    vmin = min_val if min_val != None else gdf[column].min()
    vmax = max_val if max_val != None else gdf[column].max()
    cmap = get_cmap(colormap)
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    t1 = time.time()
    for idx, row in gdf.iterrows():
        if row['geometry'].geom_type in ['Polygon', 'MultiPolygon'] and row['geometry'].area < 1e-6:
            continue
        color = sm.to_rgba(row[column]) if row[column] != None else '0.95'
        ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor=color, edgecolor='black', linewidth=0.1)
        ax_n.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor=color, edgecolor='black', linewidth=0.03)
    t2 = time.time()
    print(f'complete drawing correlation, spend {t2 - t1} seconds.')

    legend_list = [mpatches.Patch(facecolor='0.95', edgecolor='0.2', linewidth=0.2, label=lengend_label)]

    sm.set_array([])
    vcenter = round((vmin + vmax) / 2, 2)
    sm.set_clim(vmin=vmin, vmax=vmax)
    axins = ax.inset_axes([0.81, 0.33, 0.03, 0.33]) # (x, y, width, height)
    cbar = plt.colorbar(sm, cax=axins, label=cbar_label)
    cbar.set_label(cbar_label, fontsize=14, fontproperties=prop)
    cbar.set_ticks([vmin, vcenter, vmax])
    cbar.ax.set_yticklabels([f'{vmin:.2f}', f'{vcenter:.2f}', f'{vmax:.2f}'], fontproperties=prop)
    cbar.ax.tick_params(length=2, width=0.5, labelsize=14)
    cbar.outline.set_linewidth(0.5)
    leg = ax.legend(handles=legend_list, bbox_to_anchor=(0.005, 0.01), loc='lower left', fontsize=14, title='', shadow=False, fancybox=False, framealpha=0)#, prop=prop)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    t3 = time.time()
    print(f'save map to {save_path}. spend {t3 - t2} seconds.')


if __name__ == "__main__":
    __spec__ = None

    ### load data
    load_data_path = '../data_processed/load_data/load_data.json'
    with open(load_data_path, 'r') as f:
        load_data = json.load(f)

    output_dir = '../data_processed/output/fig5'
    os.makedirs(output_dir, exist_ok=True)
    output_dir_load_comp = os.path.join(output_dir, 'load_components')
    os.makedirs(output_dir_load_comp, exist_ok=True)
    output_dir_analysis = os.path.join(output_dir, 'analysis')
    os.makedirs(output_dir_analysis, exist_ok=True)
    output_dir_map = os.path.join(output_dir, 'maps')
    os.makedirs(output_dir_map, exist_ok=True)

    log = open(os.path.join(output_dir, 'log_w_wo_storage.txt'), 'w', encoding='utf-8')

    tolal_load = 0
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
                     '青海省':'qinghai', '宁夏回族自治区':'ningxia', '新疆维吾尔自治区':'xinjiang', 
                     '台湾省':'taiwan', '香港特别行政区':'Hongkong', '澳门特别行政区':'Macao'}

    flexibilities = [1, 0.9, 0.8, 0.7]
    base_load_country = [0, 0, 0, 0]
    storage_country = [np.zeros((8760-24)) for i in range(4)]
    load_met_by_power_country_w_st = [np.zeros((8760-24)) for i in range(4)]
    load_met_by_balance_country_w_st = [np.zeros((8760-24)) for i in range(4)]
    load_met_by_storage_country_w_st = [np.zeros((8760-24)) for i in range(4)]
    load_met_by_power_country_wo_st = [np.zeros((8760-24)) for i in range(4)]
    load_met_by_balance_country_wo_st = [np.zeros((8760-24)) for i in range(4)]
    solar_country = np.zeros((8760))
    wind_country = np.zeros((8760))
    for flexibility in flexibilities:
        save_dir_load_comp = os.path.join(output_dir_load_comp, f'flexibility_{flexibility}')
        os.makedirs(save_dir_load_comp, exist_ok=True)

    solar_wind['load'] = None
    solar_wind['s_w_power'] = None
    solar_wind['ba_w_st_f0.8'] = None
    solar_wind['ba_wo_st_f0.8'] = None
    solar_wind['ba_diff_f0.8'] = None
    min_ba_value = 1e5
    max_ba_value = 0
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
            load_prov = load_prov[24:]

            solar_wind.loc[solar_wind['省'] == prov_name_chi, 'load'] = load_prov.sum() # TWh

        solar_prov = solar_prov[16:-8]
        wind_prov = wind_prov[16:-8]
        power_prov = solar_prov + wind_prov

        solar_wind.loc[solar_wind['省'] == prov_name_chi, 's_w_power'] = power_prov.sum() # TWh

        if prov_name_eng not in load_data:
            continue

        print_save(f'### {prov_name_chi} ###', log)
        peak_load = load_prov.max()
        min_load = load_prov.min()
        thresh = min_load / peak_load
        for i, flexibility in enumerate(flexibilities):
            if (1 - flexibility) > thresh:
                base_load = min_load
            else:
                base_load = peak_load * (1 - flexibility)
            base_load_country[i] += base_load
            load_prov_flex = load_prov - base_load
            assert not np.any(load_prov_flex < 0)
            storage_list, load_met_by_power_list_w_st, load_met_by_balance_list_w_st, load_met_by_storage_list = balance_calculation_with_storage(
                power_prov, load_prov, base_load)
            load_met_by_power_list_wo_st, load_met_by_balance_list_wo_st = balance_calculation_without_storage(power_prov, load_prov, base_load)

            storage_list = np.array(storage_list)
            load_met_by_power_list_w_st = np.array(load_met_by_power_list_w_st)
            load_met_by_balance_list_w_st = np.array(load_met_by_balance_list_w_st)
            load_met_by_storage_list = np.array(load_met_by_storage_list)

            load_met_by_power_list_wo_st = np.array(load_met_by_power_list_wo_st)
            load_met_by_balance_list_wo_st = np.array(load_met_by_balance_list_wo_st)

            storage_country[i] += storage_list
            load_met_by_power_country_w_st[i] += load_met_by_power_list_w_st
            load_met_by_balance_country_w_st[i] += load_met_by_balance_list_w_st
            load_met_by_storage_country_w_st[i] += load_met_by_storage_list
            total_load_met_by_power_w_st = load_met_by_power_list_w_st.sum()
            total_load_met_by_balance_w_st = load_met_by_balance_list_w_st.sum()
            total_load_met_by_storage = load_met_by_storage_list.sum()

            load_met_by_power_country_wo_st[i] += load_met_by_power_list_wo_st
            load_met_by_balance_country_wo_st[i] += load_met_by_balance_list_wo_st
            total_load_met_by_power_wo_st = load_met_by_power_list_wo_st.sum()
            total_load_met_by_balance_wo_st = load_met_by_balance_list_wo_st.sum()


            print_save(f'---flexibility: {flexibility}---', log)
            print_save(f'With Storage\nload met by power: {total_load_met_by_power_w_st:.4f}, '
                f'load met by balance: {total_load_met_by_balance_w_st:.4f}, load met by storage: {total_load_met_by_storage:.4f}', log)
            print_save(f'Without Storage\nload met by power: {total_load_met_by_power_wo_st:.4f}, '
                f'load met by balance: {total_load_met_by_balance_wo_st:.4f}', log)

            # draw load component curve
            save_path = os.path.join(output_dir_load_comp, f'flexibility_{flexibility}/load_component_curve_w_st_{prov_name_eng}.png')
            # draw_load_components(load_met_by_power_list_w_st, load_met_by_balance_list_w_st, load_met_by_storage_list, base_load, save_path)
            save_path = os.path.join(output_dir_load_comp, f'flexibility_{flexibility}/load_component_curve_wo_st_{prov_name_eng}.png')
            # draw_load_components(load_met_by_power_list_wo_st, load_met_by_balance_list_wo_st, base_load=base_load, save_path=save_path)

            if flexibility == 0.8:
                solar_wind.loc[solar_wind['省'] == prov_name_chi, 'ba_w_st_f0.8'] = total_load_met_by_balance_w_st
                solar_wind.loc[solar_wind['省'] == prov_name_chi, 'ba_wo_st_f0.8'] = total_load_met_by_balance_wo_st
                solar_wind.loc[solar_wind['省'] == prov_name_chi, 'ba_diff_f0.8'] = total_load_met_by_balance_wo_st - total_load_met_by_balance_w_st

                min_ba_value = min(min_ba_value, total_load_met_by_balance_w_st, total_load_met_by_balance_wo_st)
                max_ba_value = max(max_ba_value, total_load_met_by_balance_w_st, total_load_met_by_balance_wo_st)

    
    ### 
    print_save(f'### whole country (solar-wind complementary in province level) ###', log)
    total_load_province_level = []
    storage_max_province_level = []
    load_country = load_country[24:]
    for i, (flexibility, load_met_by_balance_country_w_st_f, load_met_by_balance_country_wo_st_f) in enumerate(zip(
            flexibilities, load_met_by_balance_country_w_st, load_met_by_balance_country_wo_st)):
        total_load_met_by_balance_country_w_st = load_met_by_balance_country_w_st_f.sum()
        total_load_met_by_balance_country_wo_st = load_met_by_balance_country_wo_st_f.sum()

        total_load_dict = {}
        total_load_dict['flexibility'] = flexibility
        total_load_dict['total_load_met_by_balance_w_st'] = total_load_met_by_balance_country_w_st
        total_load_dict['total_load_met_by_balance_wo_st'] = total_load_met_by_balance_country_wo_st
        total_load_province_level.append(total_load_dict)

        print_save(f'---flexibility: {flexibility}---', log)
        print_save(f'With Storage\nload met by balance: {total_load_met_by_balance_country_w_st:.4f}', log)
        print_save(f'Without Storage\nload met by balance: {total_load_met_by_balance_country_wo_st:.4f}', log)

        # draw load component curve
        save_path = os.path.join(output_dir_load_comp, f'flexibility_{flexibility}/load_component_sum_of_provinces_w_st.png')
        draw_load_components(load_met_by_power_country_w_st[i], load_met_by_balance_country_w_st[i], load_met_by_storage_country_w_st[i], 
            base_load_country[i], save_path)
        save_path = os.path.join(output_dir_load_comp, f'flexibility_{flexibility}/load_component_sum_of_provinces_wo_st.png')
        draw_load_components(load_met_by_power_country_wo_st[i], load_met_by_balance_country_wo_st[i], base_load=base_load_country[i], save_path=save_path)


    ### statistics
    print_save('---statistics---', log)
    average_load = load_country.sum() / len(load_country)
    print_save(f'average_load: {average_load} TWh', log)

    ### 
    diff_province_country = []
    for province_level in total_load_province_level:
        diff_dict = {}
        diff_dict['flexibility'] = province_level['flexibility']
        diff_dict['diff_balance'] = round(province_level['total_load_met_by_balance_wo_st'] - province_level['total_load_met_by_balance_w_st'], 4)
        diff_dict['diff_balance_equal_hour_of_avgload'] = round(diff_dict['diff_balance'] / average_load, 4)
        diff_province_country.append(diff_dict)
    unit = {'unit': 'TWh'}
    diff_province_country.append(unit)

    # with open(os.path.join(output_dir_analysis, 'diff_balance_country.json'), 'w') as f:
    #     json.dump(diff_province_country, f)
    

    ### draw total load bar (country-level and province-level)
    total_balance_w_st = []
    total_balance_wo_st = []
    for province_level in total_load_province_level:
        total_balance_w_st.append(round(province_level['total_load_met_by_balance_w_st']))
        total_balance_wo_st.append(round(province_level['total_load_met_by_balance_wo_st']))
    save_path = os.path.join(output_dir_analysis, 'bar_balance_energy_with_flexibility.png')
    draw_bars(total_balance_w_st, total_balance_wo_st, save_path)


    ### draw diff energy bar
    diff_balance = []
    diff_balance_hour = []
    for diff_dict in diff_province_country:
        if 'flexibility' not in diff_dict:
            continue
        diff_balance.append(abs(round(diff_dict['diff_balance'])))
        diff_balance_hour.append(abs(round(diff_dict['diff_balance_equal_hour_of_avgload'])))
    save_path = os.path.join(output_dir_analysis, 'bar_balance_energy_diff_with_flexibility.png')
    draw_diff_bars_twh(diff_balance, save_path)
    save_path = os.path.join(output_dir_analysis, 'bar_balance_energy_hour_diff_with_flexibility.png')
    draw_diff_bars(diff_balance_hour, save_path, unit='hour')


    ### draw balance power map
    colormap = 'Oranges'
    lengend_label = 'No load data'
    cbar_label = 'Flexible generation (TWh)'
    save_path = os.path.join(output_dir_map, f'balance_map_w_st_f0.8_{colormap}.png')
    draw_map(solar_wind, 'ba_w_st_f0.8', min_ba_value, max_ba_value, colormap, lengend_label=lengend_label, cbar_label=cbar_label, save_path=save_path)

    save_path = os.path.join(output_dir_map, f'balance_map_wo_st_f0.8_{colormap}.png')
    draw_map(solar_wind, 'ba_wo_st_f0.8', min_ba_value, max_ba_value, colormap, lengend_label=lengend_label, cbar_label=cbar_label, save_path=save_path)

    save_path = os.path.join(output_dir_map, f'balance_diff_map_f0.8_{colormap}.png')
    draw_map(solar_wind, 'ba_diff_f0.8', colormap=colormap, lengend_label=lengend_label, cbar_label=cbar_label, save_path=save_path)

    log.close()
