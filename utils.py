
#%%
import pandas as pd
import datetime

path_csv = "video-links/driving/J Utah.csv"

df = pd.read_csv(path_csv)
pd.to_timedelta(df['duration']).sum() / datetime.timedelta(hours=1)


# %%
