#%%
import pandas as pd

#%%
def create_single_video_dataset(
    all_comments: pd.DataFrame,
    random_state: int,
    output_path: str
):
    # 1. Sample ONE VideoID using seed
    video_id = (
        all_comments['VideoID']
        .drop_duplicates()
        .sample(n=1, random_state=random_state)
        .iloc[0]
    )

    # 2. Copy dataset for that video
    single_video_comments = (
        all_comments[all_comments['VideoID'] == video_id]
        .copy()
    )

    # 3. Save
    single_video_comments.to_csv(output_path, index=False)

    # 4. Logs
    print(f"Random seed      : {random_state}")
    print(f"Selected VideoID : {video_id}")
    print(f"Total comments   : {len(single_video_comments)}")

    return single_video_comments

#%%
all_comments=pd.read_csv("../data/youtube_comments_cleaned.csv")

#%%
seeds = [1, 7, 42, 99]

for seed in seeds:
    create_single_video_dataset(
        all_comments,
        random_state=seed,
        output_path=f"../data/single_video_comments/seed_{seed}.csv"
    )

# %%
