import os
import pandas as pd

folder_path = 'image'

all_data = []

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        category = filename[:-4]  
        file_path = os.path.join(folder_path, filename)
        
        df = pd.read_csv(file_path)
        
        df['category'] = category
        
        df = df[['image', 'score', 'category']]
        
        all_data.append(df)

final_df = pd.concat(all_data, ignore_index=True)

final_df.to_csv('train.csv', index=False)

