#%%
import pandas as pd 
#%% 
top_n_comments=pd.read_csv("../data/top_n_comments.csv")
seed_1=pd.read_csv("../data/single_video_comments/seed_1.csv")
print(top_n_comments.shape)
#%%
top_n_comments.info()
#%%
seed_1.info()
# %%
top_n_comments["VideoID"].unique().shape
# %%
