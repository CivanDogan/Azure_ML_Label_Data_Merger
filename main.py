



import pandas as pd
import os
from sklearn.model_selection import train_test_split
from coolname import generate_slug

#texts = pd.read_json("input/data_for_nlp_sanitized.json", lines=True)

texts = pd.read_excel("input/data_for_nlp_sanitized.xlsx")

df = pd.DataFrame()
for file in os.listdir("./files"):
    if file.endswith(".jsonl"):
        df = pd.concat([df, pd.read_json("./files/"+file, lines=True)])


#replace tring from image_url with empty string
df['image_url'] = df['image_url'].str.replace('AmlDatastore://workspaceblobstore/Labeling/outputs/doNotDelete/tabularDataset/conversion/UX/d01d73bf-f487-d5e9-62a4-74a3a428011a/jsonlines_row_','')
df['image_url'] = df['image_url'].str.replace('.txt','')


df["image_url"] = df["image_url"].apply(lambda x: texts["full_text"][int(x)])


test_size = 0.2
train, test = train_test_split(df, test_size=test_size, random_state=42)

dataset_name = generate_slug(2)
#create a folder with the name of the dataset
os.mkdir(f"output/{dataset_name}")
df.to_json(f"output/{dataset_name}/full-{len(df)}-labeled-{dataset_name}.jsonl", orient="records", lines=True, force_ascii=False)
train.to_json(f"output/{dataset_name}/train-%{(1- test_size)*100}-{dataset_name}.jsonl", orient="records", lines=True, force_ascii=False)
test.to_json(f"output/{dataset_name}/test-%{test_size*100}-{dataset_name}.jsonl", orient="records", lines=True, force_ascii=False)








