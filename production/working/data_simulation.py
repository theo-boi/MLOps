import os, random
from PIL import Image
import numpy as np
import pandas as pd

random.seed(0)


def _formatdataframe(df, keep_lasts=-1, end= 0):
    if keep_lasts < 0:
        keep_lasts = len(df)
    
    # Mix the dataframe and keep the lasts `keep_lasts` rows
    df = df.sample(frac=1).tail(keep_lasts).reset_index()
    df = df.rename(index=lambda x: x + end - len(df))
    
    # Insert empty lines at the beginning so that it reaches the end if it does not have enough
    df = pd.concat(
        [pd.DataFrame(data=np.nan, columns=df.columns, index=range(end - len(df))), df]
    )
    return df


def dataDirToDataFrame(digits_dirs):
    data = []
    for image_dir in os.listdir(digits_dirs):
        image_dir = os.path.join(digits_dirs, image_dir)
        image_list = os.listdir(image_dir)
        for image in image_list:
            image_path = os.path.join(image_dir, image)
            image = Image.open(image_path).convert('1').resize((28, 28)) # convert to black and white
            data.append(list(image.getdata()))
    return pd.DataFrame(data, columns=['pixel' + str(i) for i in range(image.size[0] * image.size[1])])


def getSimulationData(data_descriptions):
    max_len = max(map(lambda d: d['num_samples'], data_descriptions.values()))
    dataframes = [
        _formatdataframe(
            df=df_description['source'],
            keep_lasts=df_description['num_samples'],
            end= max_len
        )
        for df_name, df_description in data_descriptions.items()
    ]
    
    # Assign the id of the dataframe in a new column "source"
    dataframes = list(map(lambda i: i[1].assign(source=i[0]), enumerate(dataframes)))

    # Generating weights
    dfs_weights = [
        list(np.full(max_len - df_description['num_samples'], -np.inf))
        +
        list(
            df_description['dist'](df_description['num_samples']) +
            df_description['noise'](df_description['num_samples'])
        )
        for i, df_description in enumerate(data_descriptions.values())
    ]

    # Add weight columns to each dataframe
    dataframes = map(lambda i: i[1].assign(weight=dfs_weights[i[0]]), enumerate(dataframes))

    # Concatenate dataframes
    df = pd.concat(dataframes)
    df['index'] = df.index
    
    # Select the row with the maximum weight for each combination of "index" and "source"
    df = df.sort_values(['weight', 'index'], ascending=[False, True])
    df = df.groupby('index').first()

    return df, dfs_weights
