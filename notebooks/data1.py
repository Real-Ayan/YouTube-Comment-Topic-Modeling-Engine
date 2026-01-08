#%%
import pandas as pd 
from datetime import datetime, timezone
#%% 
all_comments=pd.read_csv("../data/youtube_comments_cleaned.csv")
print(all_comments.shape)

# %%
all_comments.head()

# %%
all_comments.info()

# %%

all_comments["VideoID"].unique().shape
# %%
# Count comments per video and get top n videos
n=10
video_counts = all_comments['VideoID'].value_counts()
top_n_video_ids = video_counts.head(n).index

# Filter the original dataframe to keep only comments from top n videos
top_n_comments = all_comments[all_comments['VideoID'].isin(top_n_video_ids)].copy()

print(f"Original unique videos: {all_comments['VideoID'].nunique()}")
print(f"Filtered unique videos: {top_n_comments['VideoID'].nunique()}")
print(f"Total comments in top {n} videos: {len(top_n_comments)}")

# %%
top_n_comments.info()
#%%
top_n_comments.to_csv(f"../data/top_{n}_comments.csv", index=False)

# %%
# top_n_comments.drop(columns=['AuthorName','AuthorChannelID'], axis=1, inplace=True)

