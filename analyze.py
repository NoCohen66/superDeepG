#!/usr/bin/env python
import sys
import pandas as pd
import os 

def main():

    if len(sys.argv) < 2:
        print("Usage: python analyze.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)
    df['robust'] = df['robust'].astype('boolean').fillna(False)    
    df['incorrect'] = pd.to_numeric(df['incorrect'], errors='coerce').fillna(0)
    df['exec_time'] = pd.to_numeric(df['exec_time'], errors='coerce').fillna(0)
    
    def aggregate_group(group):
        overall_robust = group['robust'].all()
        avg_time = group['exec_time'].sum() if not group.empty else 0
        return pd.Series({'robust': overall_robust, 'time': avg_time})
    
    grouped = df.groupby('image_index').apply(aggregate_group, include_groups=False).reset_index()
    percent_im_robust = (grouped['robust'].sum() / len(grouped)) * 100
    average_time = grouped['time'][2:].mean()  
    #print(grouped.to_csv(index=False))
    
    print("###############################")
    print("########### - RESULTS - #############")
    print(csv_path.split("/")[-1])
    print(f"{percent_im_robust}% certified images, {grouped['robust'].sum()} on {len(grouped)} images")
    print(f"Computed in {average_time} seconds/per image")
    print("###############################")
    print("###############################")
if __name__ == "__main__":
    main()