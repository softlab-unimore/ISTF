"""
This source code is adapted and modified from
	https://github.com/bw-park/ACSSM
Authors: Byoungwoo Park, Hyungi Lee, Juho Lee (ICLR 2025)
"""
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import gzip
from itertools import chain

import numpy as np
import pandas as pd


from urllib.request import Request, urlopen

import tarfile

def extract_tar_gz(tar_gz_path, extract_dir):
    """
    Extracts a .tar.gz file to a specified directory.
    """
    
    if os.path.exists(os.path.join(extract_dir, 'pub12')):
        return
    
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_gz_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)
    print(f"Extracted {tar_gz_path} to {extract_dir}")


def download_url(url, root, filename=None):
    os.makedirs(root, exist_ok=True)
    if filename is None:
        filename = os.path.basename(url)
    filepath = os.path.join(root, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading {url} to {filepath}")
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(req) as response, open(filepath, 'wb') as out_file:
            out_file.write(response.read())
    else:
        print(f"File already exists at {filepath}")
    return filepath


def download_ushcn(data_path):
    os.makedirs(data_path, exist_ok=True)
    url = 'https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-7b1e0d7f2fc3c43-20180727T175547656'
    data_name = "ushcn_daily.tar.gz"
    download_url(url, data_path, data_name)
    file_name = os.path.join(data_path, data_name)
    extract_tar_gz(file_name, data_path)


def to_pandas(state_dir, target_dir):
    name = state_dir[:-4] + '.csv'
    output_file = os.path.join(target_dir, name)

    if not os.path.exists(output_file):
        sep_list = [0, 6, 10, 12, 16]
        for day in range(31):
            sep_list = sep_list + [21 + day * 8, 22 + day * 8, 23 + day * 8, 24 + day * 8]

        columns = ['COOP_ID', 'YEAR', 'MONTH', 'ELEMENT']
        values_list = list(chain.from_iterable(("VALUE-" + str(i + 1), "MFLAG-" +
                                                str(i + 1), "QFLAG-" + str(i + 1), "SFLAG-" + str(i + 1)) for i in range(31)))
        columns += values_list

        df_list = []
        with gzip.open(os.path.join(target_dir, state_dir), 'rt') as f:
            for line in f:
                line = line.strip()
                nl = [line[sep_list[i]:sep_list[i + 1]] for i in range(len(sep_list) - 1)]
                df_list.append(nl)

        df = pd.DataFrame(df_list, columns=columns)
        val_cols = [s for s in columns if "VALUE" in s]

        df[val_cols] = df[val_cols].astype(np.float32)

        df.replace(r'\s+', np.nan, regex=True, inplace=True)
        df.replace(-9999, np.nan, inplace=True)

        df_m = df.melt(id_vars=["COOP_ID", "YEAR", "MONTH", "ELEMENT"])
        df_m[["TYPE", "DAY"]] = df_m.variable.str.split(pat="-", expand=True)

        df_n = df_m[["COOP_ID", "YEAR", "MONTH",
                     "DAY", "ELEMENT", "TYPE", "value"]].copy()

        df_p = df_n.pivot_table(values='value', index=[
            "COOP_ID", "YEAR", "MONTH", "DAY", "ELEMENT"], columns="TYPE", aggfunc="first")
        df_p.reset_index(inplace=True)

        df_q = df_p[["COOP_ID", "YEAR", "MONTH", "DAY",
                     "ELEMENT", "MFLAG", "QFLAG", "SFLAG", "VALUE"]]

        print(f"Saving {output_file}")
        df_q.to_csv(output_file, index=False)
        # Number of non missing
        # meas_tot = df.shape[0]*31
        # na_meas = df[val_cols].isna().sum().sum()


def convert_all_to_pandas(input_dir, output_dir):
    list_dir = os.listdir(input_dir)
    txt_list_dir = [s for s in list_dir if s.startswith('state') and s.endswith('.txt.gz')]
    state_list_dir = [s for s in txt_list_dir if "state" in s]

    for i, state_dir in enumerate(state_list_dir):
        print(f'Computing State {i}: {state_dir}')
        to_pandas(state_dir, output_dir)


def merge_dfs(input_dir, output_dir, keyword):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "daily_merged.csv")
    if not os.path.exists(output_file):
        df_list = os.listdir(input_dir)
        csv_list = [s for s in df_list if '.csv' in s]
        keyword_csv_list = [s for s in csv_list if keyword in s]

        df_list = []
        for keyword_csv in keyword_csv_list:
            print(f"Loading dataframe for keyword : {keyword_csv[:-4]}")
            df_temp = pd.read_csv(os.path.join(input_dir, keyword_csv), low_memory=False)
            # df_temp.insert(0, "UNIQUE_ID", keyword_csv[-7:-4])
            df_list.append(df_temp)
        print("All dataframes are loaded")
        # Merge all datasets:
        print("Concat all ...")
        df = pd.concat(df_list)
        df.to_csv(output_file, index=False)


def clean(input_dir, output_dir, start_year=1990, end_year=1993):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"cleaned_df_{start_year}_{end_year}.csv")
    if not os.path.exists(output_file):
        df = pd.read_csv(os.path.join(input_dir, "daily_merged.csv"), low_memory=False)
        print(f"Loaded df. Number of observations : {df.shape[0]}")
        # Remove NaNs
        df.drop(df.loc[df.VALUE.isna()].index, inplace=True)

        # Remove values with bad quality flag.
        qflags = ["D", "G", "I", "K", "L", "M",
                  "N", "O", "R", "S", "T", "W", "X", "Z"]
        df.drop(df.loc[df.QFLAG.isin(qflags)].index, inplace=True)
        print(f"Removed bad quality flags. Number of observations {df.shape[0]}")

        # Drop centers which observations end before 1994
        gp_id_year = df.groupby("COOP_ID")["YEAR"]
        print(f"Drop centers which observations end before 1994")
        coop_list = list(gp_id_year.max().loc[gp_id_year.max() >= end_year + 1].index)
        df.drop(df.loc[~df.COOP_ID.isin(coop_list)].index, inplace=True)

        # Drop center which observations start after 1990
        gp_id_year = df.groupby("COOP_ID")["YEAR"]
        crop_list = list(gp_id_year.min().loc[gp_id_year.min() <= start_year].index)
        df.drop(df.loc[~df.COOP_ID.isin(crop_list)].index, inplace=True)

        # Crop the observations before 1950 and after 2001.
        df = df.loc[df.YEAR >= 1950].copy()
        df = df.loc[df.YEAR <= 2000].copy()

        print(f"Number of kept centers : {df.COOP_ID.nunique()}")
        print(
            f"Number of observations / center : {df.shape[0] / df.COOP_ID.nunique()}")
        print(f"Number of days : {50 * 365}")
        print(f"Number of observations : {df.shape[0]}")

        cols = ['COOP_ID', 'YEAR', 'MONTH', 'DAY', 'ELEMENT', 'VALUE']
        df[cols].to_csv(output_file, index=False)


# new code
def ushcn_to_timeseries(input_dir, output_dir, start_year, end_year):
    # Read the input CSV file
    input_file = os.path.join(input_dir, f"cleaned_df_{start_year}_{end_year}.csv")
    data = pd.read_csv(input_file, low_memory=False)

    # Filter data for the specified year range
    data = data[(data['YEAR'] >= start_year) & (data['YEAR'] <= end_year)].copy()

    # Convert year, month, and day columns to a single datetime column
    data['DATE'] = data.apply(lambda x: pd.Timestamp(year=x['YEAR'], month=x['MONTH'], day=x['DAY']), axis=1)

    # Remove the original date columns
    data.drop(columns=['YEAR', 'MONTH', 'DAY'], inplace=True)

    # Define the output file path
    output_file = os.path.join(output_dir, f"pivot_{start_year}_{end_year}.csv")

    # Create a pivot table
    pivot_data = data.pivot_table(values='VALUE', index=['COOP_ID', 'DATE'], columns='ELEMENT', aggfunc='first')
    pivot_data.reset_index(inplace=True)

    # Create TAVG feature
    pivot_data['TAVG'] = pivot_data[['TMIN', 'TMAX']].mean(skipna=False, axis=1)

    # Save the pivot table to a CSV file
    pivot_data.to_csv(output_file, index=False)

    print(f"Processed data saved to {output_file}")


def extract_stations_spatial_data(input_dir, output_dir):
    # stations_txt = os.path.join(input_dir, 'ushcn-v2.5-stations.txt')
    stations_txt = os.path.join(input_dir, 'ushcn-stations.txt')
    df = pd.read_fwf(
        stations_txt,
        header=None,
        names=['station_id', 'latitude', 'longitude', 'elevation', 'state', 'station_name', 'unknown1', 'unknown2',
               'unknown3', 'timezone'],
      # colspecs=[(0, 11), (11, 20), (20, 30), (30, 37), (37, 40), (40, 71), (71, 78), (78, 85), (85, 92), (92, 94)],
        colspecs=[(0,  6), ( 6, 15), (15, 25), (25, 32), (32, 35), (35, 66), (66, 73), (73, 80), (80, 87), (87, 89)],
        converters={'station_name': lambda x: x.strip()}
    )
    # output_file = os.path.join(output_dir, 'ushcn-v2.5-stations.csv')
    output_file = os.path.join(output_dir, 'ushcn-stations.csv')
    df.to_csv(output_file, index=False)


def augment_with_spatial_data(input_dir, output_dir, start_year, end_year):
    # Read the pivot data and spatial data
    pivot_file = os.path.join(input_dir, f"pivot_{start_year}_{end_year}.csv")
    # stations_file = os.path.join(input_dir, 'ushcn-v2.5-stations.csv')
    stations_file = os.path.join(input_dir, 'ushcn-stations.csv')

    pivot_data = pd.read_csv(pivot_file, low_memory=False)
    stations_data = pd.read_csv(stations_file, low_memory=False)

    # Extract COOP ID from station_id and set it as index for efficient mapping
    # stations_data['COOP_ID'] = stations_data['station_id'].str[3:].astype(int)
    stations_data['COOP_ID'] = stations_data['station_id'].astype(int)
    stations_data.set_index('COOP_ID', inplace=True)

    # Add spatial data to pivot data
    pivot_data['LATITUDE'] = pivot_data['COOP_ID'].map(stations_data['latitude'])
    pivot_data['LONGITUDE'] = pivot_data['COOP_ID'].map(stations_data['longitude'])
    pivot_data['ELEVATION'] = pivot_data['COOP_ID'].map(stations_data['elevation'])

    # Save the augmented data
    output_file = os.path.join(output_dir, f"pivot_{start_year}_{end_year}_spatial.csv")
    pivot_data.to_csv(output_file, index=False)

    print(f"Augmented data saved to {output_file}")


def download_and_process_ushcn(file_path):
    raw_path = os.path.join(file_path, 'raw')
    processed_path = os.path.join(file_path, 'processed')
    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(processed_path, exist_ok=True)

    start_year = 1990
    end_year = 1993

    # downloads daily state files from ushcn webpage
    download_ushcn(raw_path)

    # converts .txt files to .csv
    raw_path = os.path.join(raw_path + "/pub12/ushcn_daily")
    convert_all_to_pandas(input_dir=raw_path, output_dir=raw_path)

    # reads the individual state .csv files and merges it to a single file named "daily_merged.csv"
    merge_dfs(input_dir=raw_path, output_dir=processed_path, keyword='state')

    # cleans daily_merged.csv
    clean(input_dir=processed_path, output_dir=processed_path, start_year=start_year, end_year=end_year)

    # create multivariate time series dataset
    ushcn_to_timeseries(input_dir=processed_path, output_dir=processed_path, start_year=start_year, end_year=end_year)

    # extract spatial information
    extract_stations_spatial_data(input_dir=raw_path, output_dir=processed_path)

    # extend multivariate time series dataset with spatial information
    augment_with_spatial_data(input_dir=processed_path, output_dir=file_path, start_year=start_year,
                              end_year=end_year)


if __name__ == '__main__':
    download_and_process_ushcn('data/USHCN')
