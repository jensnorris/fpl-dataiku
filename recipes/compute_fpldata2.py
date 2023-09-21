# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import requests
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import warnings
from sklearn import metrics
import pickle


# Read recipe inputs
fpldata = dataiku.Dataset("fpldata")
fpldata_df = fpldata.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

def get_player_X_and_y(name):
    df_player = df.loc[df['name'] == name]
    df_actual_game_attributes = df_player.loc[:, ['round', 'kickoff_time', 'was_home', 'difficulty', 'opponent_team', 'playing_in', 'total_points']]
    df_actual_game_attributes.rename({'total_points': 'y'}, axis=1, inplace=True)
    df_rolling_mean_attributes = df_player.drop(['round', 'name', 'element', 'kickoff_time', 'difficulty', 'opponent_team', 'transfers_balance', 'value', 'was_home', 'playing_in'], axis=1).shift(1).rolling(3).mean()
    player_df_stats_3past_games = df_actual_game_attributes.join(df_rolling_mean_attributes).dropna()
    player_df_stats_3past_games.rename({'was_home': 'is_home'}, axis=1, inplace=True)
    return player_df_stats_3past_games

def get_all_players_df():
    df_Xy = pd.DataFrame()
    for p in fpldata_df.name.unique().tolist():
        Xy = get_player_X_and_y(p)
        df_Xy = df_Xy.append(Xy)
    return df_Xy

df_Xy = get_all_players_df()

fpldata2_df = df_Xy # For this sample code, simply copy input to output

# Write recipe outputs
fpldata2 = dataiku.Dataset("fpldata2")
fpldata2.write_with_schema(fpldata2_df)