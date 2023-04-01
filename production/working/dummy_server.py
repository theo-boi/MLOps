import os, sys, json
import numpy as np
import pandas as pd
from tqdm import tqdm

class DummyServer:
    def __init__(self, input_df, storage):
        self.input = input_df
        self.storage = storage
        
    def setup(self):
        data_generator = self.input.iterrows()
        for (id1, s1) in data_generator:
            if not sum(map(lambda i: 'pixel' in i, s1.index)):
                print(f"{id1} ignored: data is malformed.")
            data_request = s1.to_frame().T
            sample_request = {'dataframe_split': data_request.to_dict(orient='split')}
            with open(os.path.join(self.storage, f"{id1}.json"), "w") as output:
                json.dump(sample_request, output, indent=4)
    
    def do_GET(self):
        for file in tqdm(os.listdir(self.storage)):
            try:
                if file[0] == '.': continue
                with open(os.path.join(self.storage, file), 'r') as f:
                    sample_request = json.load(f)
            except ValueError:
                print(f"{file} is not JSON")
                continue
            yield json.dumps(sample_request)
