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

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.patches import Circle, Arc, PathPatch
from matplotlib.path import Path
from matplotlib.font_manager import FontProperties


def draw_stack_bar(x, y1, y2, save_path):
    font_path = '/System/Library/Fonts/Helvetica.ttc'
    prop = FontProperties(fname=font_path)

    plt.figure(figsize=(50, 8.5))
    width=0.8
    plt.bar(x, y1, width=width, label='Solar Generation', color='darkred', alpha=1) # '#F8B072' '#F898CB', '#FF8C00', '#EC3E31'
    plt.bar(x, y2, width=width, bottom=y1, label='Wind Generation', color='darkgreen', alpha=1) # '#4F99C9' '#4DAF4A', '#699ECA'

    plt.ylabel('Estimated generation (GWh)', fontproperties=prop, fontsize=38)
    # plt.legend(fontsize=8)

    plt.xticks(fontproperties=prop)
    plt.yticks(fontproperties=prop)
    plt.tick_params(axis='both', labelsize=38)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.margins(x=0.01)
    ax.tick_params(axis='y', length=0)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return


if __name__ == "__main__":

    output_dir = f'../output/fig1'
    os.makedirs(output_dir, exist_ok=True)

    statistics_path = os.path.join(output_dir, 'statistics.json')
    if os.path.exists(statistics_path):
        with open(statistics_path, 'r') as f:
            statistics = json.load(f)
        print(f'load {statistics_path} complete!')
    else:
        solar_wind_path = '../data/solar_wind_aggregation/solar_wind_province_level.geojson'
        solar_wind = gpd.read_file(solar_wind_path)
        print('read solar_wind file completed.')

        level2property = {'省': '省', '市': '地名', '县': '地名'}
        level = '省'
        county_boundaries_path = f'../data/China_boundary_shps/2021年{level}矢量.shp'
        print(f'reading {county_boundaries_path} ...')
        county_data = gpd.read_file(county_boundaries_path)
        print('complete!')
        province_chi = [prov for prov in county_data[level2property[level]] if prov != '中朝共有']

        province_chi2eng = {'北京市':'beijing', '天津市':'tianjin', '河北省':'hebei', '山西省':'shanxi', '内蒙古自治区':'Inner Mongolia', 
                         '辽宁省':'liaoning', '吉林省':'jilin', '黑龙江省':'heilongjiang', '上海市': 'shanghai', '江苏省':'jiangsu',
                         '浙江省':'zhejiang', '安徽省':'anhui', '福建省':'fujian', '江西省':'jiangxi', '山东省':'shandong', '河南省':'henan',
                         '湖北省':'hubei', '湖南省':'hunan', '广东省':'guangdong', '广西壮族自治区':'guangxi', '海南省':'hainan', '重庆市':'chongqing',
                         '四川省':'sichuan', '贵州省':'guizhou', '云南省':'yunnan','西藏自治区':'Tibet', '陕西省':'shaanxi', '甘肃省':'gansu',
                         '青海省':'qinghai', '宁夏回族自治区':'ningxia', '新疆维吾尔自治区':'xinjiang',
                         '台湾省': 'taiwan', '香港特别行政区': 'HongKong', '澳门特别行政区': 'Macao'}
        province_eng2short = {'beijing': 'BJ', 'tianjin': 'TJ', 'hebei': 'HE', 'shanxi': 'SX', 'Inner Mongolia': 'IM',
            'liaoning': 'LN', 'jilin': 'JL', 'heilongjiang': 'HL', 'shanghai': 'SH', 'jiangsu': 'JS',
            'zhejiang': 'ZJ', 'anhui': 'AH', 'fujian': 'FJ', 'jiangxi': 'JX', 'shandong': 'SD', 'henan': 'HA',
            'hubei': 'HB', 'hunan': 'HN', 'guangdong': 'GD', 'guangxi': 'GX', 'hainan': 'HI', 'chongqing': 'CQ',
            'sichuan': 'SC', 'guizhou': 'GZ', 'yunnan': 'YN', 'Tibet': 'XZ', 'shaanxi': 'SN', 'gansu': 'GS',
            'qinghai': 'QH', 'ningxia': 'NX', 'xinjiang': 'XJ',
            'taiwan': 'TW', 'HongKong': 'HK', 'Macao': 'MO'}


        # solar polygon shapefile
        solar_path = '../data/installation_shps/solar/Qinghai.shp'
        print(f'reading {solar_path} ...')
        solar_data = gpd.read_file(solar_path)
        print('complete!')
        proj4_string = ('+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs')
        solar_data['area_py'] = solar_data.to_crs(proj4_string)['geometry'].area
        solar_data['count'] = [1 for i in range(solar_data.shape[0])]

        wind_path = '../data/installation_shps/wind/Qinghai.shp'
        print(f'reading {wind_path} ...')
        wind_data = gpd.read_file(wind_path, encoding='gbk')
        print('complete!')
        wind_data['count'] = [1 for i in range(wind_data.shape[0])]


        print(f'aggregate solar area and count by {level} ...')
        merged_data = gpd.sjoin(solar_data, county_data, op='within')
        solar_summary = merged_data.groupby(level2property[level]).agg(total_area=('area_py', 'sum'), 
            count=('count', 'sum')).reset_index()
        print('complete!')

        print(f'aggregate wind count by {level} ...')
        merged_data = gpd.sjoin(wind_data, county_data, op='within')
        wind_summary = merged_data.groupby(level2property[level]).agg(count=('count', 'sum')).reset_index()
        print('complete!')

        # aggragate statistics
        statistics = {}
        for province, province_eng in province_chi2eng.items():
            item = {}
            # item['engname'] = province_eng
            item['shortname'] = province_eng2short[province_eng]
            item['solar_area'] = float(solar_summary[solar_summary['省'] == province]['total_area'].iloc[0]) if province in np.array(solar_summary['省']) else 0
            item['solar_count'] = float(solar_summary[solar_summary['省'] == province]['count'].iloc[0]) if province in np.array(solar_summary['省']) else 0
            item['solar_power'] = solar_wind[solar_wind['省'] == province]['power_sum_solar'].iloc[0] # unit: MW
            item['wind_count'] = float(wind_summary[wind_summary['省'] == province]['count'].iloc[0]) if province in np.array(wind_summary['省']) else 0
            item['wind_power'] = solar_wind[solar_wind['省'] == province]['power_sum_wind'].iloc[0] # unit: MW
            statistics[province_eng] = item

        save_path = os.path.join(output_dir, 'statistics.json')
        with open(save_path, 'w') as f:
            json.dump(statistics, f)


    provinces = []
    solar_areas = []
    solar_counts = []
    solar_powers = []
    wind_counts = []
    wind_powers = []
    for province, item in statistics.items():
        provinces.append(item['shortname'])
        solar_areas.append(item['solar_area'])
        solar_counts.append(item['solar_count'])
        solar_powers.append(item['solar_power']/1e3) # MW -> GW
        wind_counts.append(item['wind_count'])
        wind_powers.append(item['wind_power']/1e3) # MW -> GW

    # save_path = os.path.join(output_dir, 'bar_solar_wind_power.png')
    # draw_stack_bar(provinces, solar_powers, wind_powers, save_path=save_path)

    total_powers = [solar + wind for solar, wind in zip(solar_powers, wind_powers)]
    sorted_data = sorted(zip(provinces, solar_powers, wind_powers, total_powers), key=lambda x: x[3], reverse=True)
    sorted_provinces, sorted_solar_powers, sorted_wind_powers, _ = zip(*sorted_data)
    save_path = os.path.join(output_dir, 'bar_solar_wind_power_sorted.png')
    draw_stack_bar(sorted_provinces, sorted_solar_powers, sorted_wind_powers, save_path=save_path)

