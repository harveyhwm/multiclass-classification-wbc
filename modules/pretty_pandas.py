import pandas as pd
import numpy as np
import seaborn as sns
import json
from collections import defaultdict

import dataframe_image # use this package to help print out our new styles

# set of predefined RGB colors
PALETTES_RGB = {
    'yellow':[252,239,166],
    'light_yellow':[253,247,211],
    'green':[122,188,129],
    'dark_green':[61,94,64],
    'red':[231,114,112],
    'white':[255,255,255],
    'blue':[101,147,194],
    'dark_blue':[50,73,97],
    'grey':[144,144,148],
    'sns_blue':[13,29,85],
    'sns_yellow':[255,255,221],
    'sns_green':[103,182,193]
}
PALETTES_HEX = {}

# extract the hex value from a given color and round accordingly, ensuring length==2
def make_hex_color(s, round='nearest'):
    if round=='up':
        s_round = np.ceil(s,0)
    elif round=='down':
        s_round = np.floor(s,0)
    else:
        s_round = np.round(s,0)
    return ('0'+hex(int(s_round))[2:].upper())[-2:]

# make a full hex color from 3 RGB channels
def rgb_to_hex(channels, round='nearest'):
    return '#'+(''.join([make_hex_color(c, round) for c in channels]))

# use our helpers to populate hex code dict (we will want to speak generally in hex for this work)
for i in PALETTES_RGB.keys():
    PALETTES_HEX[i] = '#'+(''.join([make_hex_color(color) for color in PALETTES_RGB[i]]))

# make a range of evenly spaced floats of a given min, max and length
def divide_range(mymin, mymax, size, thresholds=None):
    return [mymin+(k*(mymax-mymin)/(size-1)) for k in range(size)]
    # np.arange(mymin,mymax+(1/(size-1)),(1/(size-1))) # alternative way

def make_quantiles(values, n, mn, mx, spacing='relative'):
    if type(n)==list: n=len(n)
    if spacing == 'even':  # evenly distribute the color palette ignoring the magnitude of the values
        return [np.floor((n-1)*((values<=v).mean()+(values<v).mean())/2) for v in values]
    elif spacing == 'relative':  # factor in the megnitude of the values when making the visuals (default)
        return [np.maximum(0,np.minimum(int((n-1)*(v-mn)/(mx-mn)),n-2)) for v in values] # prevent negative values

# get RGB colors from hex if we want to go the other way
def get_rgb_colors(c):
    if c in PALETTES_RGB:
        return PALETTES_RGB[c]
    else:
        c = c.replace('#','')
        n = [c[i*int(len(c)/3):(i+1)*int(len(c)/3)] for i in range(3)]
        if len(c)==3: n = [s+s for s in n]
        return [int(n,16) for n in n]

# generates an RGB color value from a given float, based on its distance from defined min/max values and their associated rgb colors
def generate_color(value, thresholds, colors):
    (min,max) = thresholds
    (min_color,max_color) = colors
    diff = [min_color[i]-max_color[i] for i in range(3)]
    return [min_color[j]-(diff[j]*(value-min)/(max-min)) for j in range(3)]

def luminosity(v):
    return (0.2126*v[0]+0.7152*v[1]+0.0722*v[2])

def type_format(data,val,number):
    if number in [None,'abs']:
        return max(np.min(data),min(np.max(data),val))
    elif number=='pct':
        return np.quantile(data,val/100)

def run_transform(data,funcs):
    for f in funcs:
        if f==np.log:
            data = np.abs(data)
        try:
            data = f(data)
        except:
            data = f(np.abs(data))
        data = data.replace(np.nan, 0)
    return data

def apply_colors_legacy(col, palette=['yellow', 'green'], default_fill_color='#FFF', default_text_color='#000', type='shade', rows=None, columns=None, mymin=None, mymax=None):
    # by default, use column-wise min and max if nothing is provided
    if mymax is None: mymin, mymax = min(col.values), max(col.values)
    
    # to prevent a divide by zero later on - max must always be greater than min
    if mymax==mymin: mymax=mymin+1
    palette = [get_rgb_colors(p) for p in palette]
    
    if len(palette) == 1:
        # if the palette length is just 1 we just apply it globally - the trivial case
        rgb_vals = [palette[0] for c in col.values]
    else:
        # if the palette length is greater than 1, we assign each value a bespoke color based on its position in the full range
        thresholds = divide_range(mymin, mymax, len(palette))
        # print(thresholds)
        quantiles = make_quantiles(col.values, palette, mymin, mymax)
        # print(quantiles)
        rgb_vals = [generate_color(c, thresholds[q:q+2], palette[q:q+2]) for c,q in zip(col.values, quantiles)]

    def filter_cells(inputs, default=''):  
        if (columns is not None):
            inputs = [inputs[j] if (mymin <= col.values[j] <= mymax) and (col.name in columns) else default for j in range(len(col.values))]
        if (rows is not None):
            inputs = [inputs[j] if (mymin <= col.values[j] <= mymax) and (j in rows) else default for j in range(len(col.values))]
        return inputs
    
    if type == 'shade':
        res = ['background-color: #'+(''.join([make_hex_color(c) for c in v])) for v in rgb_vals]
        default = 'background-color: '+default_fill_color
        return filter_cells(res, default)
    elif type == 'text_shade':
        tx = ['color: '+('#000' if luminosity(v)>=100 else '#FFF') for v in rgb_vals]
        default = 'color: '+default_text_color
        return filter_cells(tx, default)
    else:
        return ['' for c in col.values]

def pretty_pandas_legacy(df, fill_palette=['yellow','green'], rows=None, columns=None, index='show', group=None, font_size=None, header_size=None,
                  default_fill_color='#FFF', default_text_color='#000', bg='white', mymin=None, mymax=None):
    """Generate efficient dataframe styling with fully customizable inputs.

    Keyword arguments:
    todo
    """
    sdf = df.style
    rows_all,columns_all = list(df.index),list(df.columns)
    if mymin is None: mymin=np.min(df.values)
    if mymax is None: mymax=np.max(df.values)

    if index=='hide': sdf.hide_index()
    if header_size is None: header_size=font_size
    if type(fill_palette[0]) != list: fill_palette=[fill_palette] 
    
    for palette in fill_palette:
        row_subset = rows_all if rows is None else [r for r in rows if r in rows_all]
        row_index_subset = [rows_all.index(r) for r in row_subset]
        col_subset = columns_all if columns is None else [c for c in columns if c in columns_all]
        d = df.loc[row_subset,col_subset]
        mymin = max(mymin, np.min(d.values)) if group is None else None
        mymax = min(mymax, np.max(d.values)) if group is None else None
        sdf.apply(apply_colors_legacy, palette=palette, default_fill_color=default_fill_color, default_text_color=default_text_color,
                  type='shade', rows=row_index_subset, columns=col_subset, mymin=mymin, mymax=mymax, axis=0)
        sdf.apply(apply_colors_legacy, palette=palette, default_fill_color=default_fill_color, default_text_color=default_text_color,
                  type='text_shade', rows=row_index_subset, columns=col_subset, mymin=mymin, mymax=mymax, axis=0)

    return sdf.format('{:.3f}').set_table_styles([{'selector':'tr','props':[('background-color',bg+' !important')]}])

    # sdf.set_properties(**{'font-size': str(font_size)+'pt'})
    # .set_table_styles([{'selector': 'th', 'props': [('font-size', str(22)+'pt !important')]}])

def apply_colors(col, value_transform=None, default_fill_color='#FFF', default_text_color='#000', default_border='', default_fill_text_colors=['#000','#FFF'],
                 thresholds=None, fill_palette=None, text_palette=None, rows=None, columns=None, mymin=None, mymax=None):

    if value_transform is not None:
        col = run_transform(col,value_transform)

    fill_palette = [None if f is None else list(get_rgb_colors(p) for p in f) for f in fill_palette]
    text_palette = [None if t is None else list(get_rgb_colors(p) for p in t) for t in text_palette]
    rgb_fill_vals,rgb_text_vals,fill_styles,default_text_styles,active_text_styles = [],[],[],[],[]
    default = 'background-color: '+default_fill_color+'; color: '+default_text_color+'; border: '+default_border
    styles = [default for j in range(len(col.values))]
    text_styles = ['' for j in range(len(col.values))]
    
    for i in range(len(fill_palette)):
        if fill_palette[i] is not None:
            if len(fill_palette[i]) == 1: # if the palette length is just 1 we just apply it globally - the trivial case
                rgb_fill_vals += [[fill_palette[i][0] for c in col.values]]
            else: # if the palette length is greater than 1, we assign each value a bespoke color based on its position in the full range
                fill_thresholds = divide_range(mymin[i], mymax[i], len(fill_palette[i]), thresholds)
                fill_quantiles = make_quantiles(col.values, fill_palette[i], mymin[i], mymax[i])
                rgb_fill_vals += [[generate_color(c, fill_thresholds[q:q+2], fill_palette[i][q:q+2]) for c,q in zip(col.values, fill_quantiles)]]
        else:
            rgb_fill_vals += [[None for c in col.values]]

        if text_palette[i] is not None:
            if len(text_palette[i]) == 1:
                rgb_text_vals += [[text_palette[i][0] for c in col.values]]
            else:
                text_thresholds = divide_range(mymin[i], mymax[i], len(text_palette[i]), thresholds)
                text_quantiles = make_quantiles(col.values, text_palette[i], mymin[i], mymax[i])
                rgb_text_vals += [[generate_color(c, text_thresholds[q:q+2], text_palette[i][q:q+2]) for c,q in zip(col.values, text_quantiles)]]
        else:
            rgb_text_vals += [[None for c in col.values]]

        fill_styles += [['background-color: '+('' if fill_palette[i] is None else '#'+''.join([make_hex_color(c) for c in v])) for v in rgb_fill_vals[-1]]]
        default_text_styles += [['color: '+('' if fill_palette[i] is None else (default_fill_text_colors[0] if luminosity(v)>=100 else default_fill_text_colors[1])) for v in rgb_fill_vals[-1]]]
        text_styles = ['color: #'+(text_styles[j] if text_palette[i] is None else ''.join([make_hex_color(c) for c in rgb_text_vals[-1][j]])) for j in range(len(col.values))]

        styles = ['; '.join([fill_styles[i][j],default_text_styles[i][j],text_styles[j]]) if (mymin[i] <= col.values[j] <= mymax[i]) and
                  (col.name in columns[i]) and (j in rows[i]) else styles[j] for j in range(len(col.values))]
    return styles

def pretty_pandas(df, fill_palette=None, text_palette=None, rows=None, columns=None, index='show', group=None, font_size=None,
                  precision=3, thresholds=None, header_size=None, default_fill_color='#FFF', default_text_color='#000', default_border='',
                  default_fill_text_colors=['#000','#FFF'], bg='white', mymin=None, mymax=None, value_transform=None, configs=None):
    """Generate efficient dataframe styling with fully customizable inputs.
    Keyword arguments:

    todo
    """
    sdf = df.style
    rows_all,columns_all = list(df.index),list(df.columns)

    def absent():
        return None
    if configs is not None:
        configs_default = [defaultdict(absent) for c in configs]
        for c in range(len(configs)):
            for i in configs[c].keys():
                configs_default[c][i] = configs[c][i]
        fill_palette = [c['fill_palette'] for c in configs_default]
        text_palette = [c['text_palette'] for c in configs_default]
        rows = [c['rows'] if c['rows'] is not None else rows_all for c in configs_default]
        row_indices = [list(rows_all.index(i) for i in r) for r in rows]
        columns = [c['columns'] if c['columns'] is not None else columns_all for c in configs_default]
        mymin, mymax = [],[]
        for i,c in enumerate(configs_default):
            # if value_transform is None:
            df_subset = df.loc[rows[i],columns[i]]
            if value_transform is not None:
                if type(value_transform) is not list: value_transform = [value_transform]
                df_subset = run_transform(df_subset,value_transform)
            mymin.append(type_format(df_subset.values,c['mymin'],c['number']) if c['mymin'] is not None else np.min(df_subset.values))
            mymax.append(type_format(df_subset.values,c['mymax'],c['number']) if c['mymax'] is not None else np.max(df_subset.values))
            # return df_subset,mymin,mymax
        # mymax = [max(mymin[m]+1,mymax[m]) for m in range(len(mymax))] # to prevent any divide by zero later on

    else:
        if mymin is None: mymin=np.min(df.values)
        if mymax is None: mymax=np.max(df.values)
        if rows is None:
            rows = rows_all
        row_indices = [rows.index(r) for r in rows] #[list(rows.index(i) for i in r) for r in rows]
        if columns is None: columns = columns_all
        mymin,mymax,rows,columns = [mymin],[mymax],[rows],[columns]

    if index=='hide': sdf.hide_index()
    if header_size is None: header_size=font_size

    sdf.apply(apply_colors, value_transform=value_transform, default_fill_color=default_fill_color, default_text_color=default_text_color,
              default_fill_text_colors=default_fill_text_colors, thresholds=thresholds, default_border=default_border,
              fill_palette=fill_palette, text_palette=text_palette, rows=row_indices, columns=columns, mymin=mymin, mymax=mymax, axis=0)

    return sdf.format('{:.'+str(precision)+'f}').set_table_styles([{'selector':'tr','props':[('background-color',bg+' !important')]}])

    # sdf.set_properties(**{'font-size': str(font_size)+'pt'})
    # .set_table_styles([{'selector': 'th', 'props': [('font-size', str(22)+'pt !important')]}])

def make_palette(*args,number='pct',palette=['white','red','yellow','green','blue'],**kwargs):
    configs = []
    if len(args)==2 or len(args)>=len(palette):
        configs.extend([
            {'fill_palette': palette[:1], 'mymax': args[0], 'number': number},
            {'fill_palette': palette[-1:], 'mymin': args[-1], 'number': number}
        ])
    else:
        if number=='pct':
            args = [0]+list(args)+[100]
        else:
            args = [-np.inf]+list(args)+[np.inf]
    for i in range(len(args)-1):
        configs.append(
            {
                'fill_palette': palette if len(args)==2 else palette[i:i+2],
                #'text_palette': palette if len(args)==2 else palette[i:i+2],
                'mymin': args[i],
                'mymax': args[i+1],
                'number': number
            }
        )
    for c in configs:
        for k in kwargs.keys(): c[k]=kwargs[k]
    return configs