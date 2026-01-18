import pandas as pd
def load_data(path):
    df=pd.read_csv(path,encoding="latin-1")
    # Keep only label and email text columns
    df=df[['v1','v2']]
    df.columns=['label','text']
    df.dropna(inplace=True)#removing missing emails
    df['label']=df['label'].map({'ham':0,'spam':1})# Convert labels to numeric
    return df