import pandas as pd

def clean_df(df, y):
    rows, cols = df.shape
    columns = list(df.columns)
    columns.remove(y)

    print("--------------------------------")

    for col in columns:
        df_list = df[col].tolist()
        test_item = str(df_list[0])
        
        if test_item.isalpha():
            pass
        else:
            if check_if_conti(df_list):
                print("Filling column " + col + " with mean of the same column")
                df[col].fillna((df[col].mean()), inplace=True)
            else:
                print("Filling column " + col + " with mode of the same column")
                df[col].fillna(df[col].mode()[0], inplace=True)
            
    return df

def check_if_conti(df_list):
    unique_count = len(set(df_list))
    length = len(df_list)

    distribution = (unique_count / length) * 100

    if distribution > 10:
        return True
    
    return False