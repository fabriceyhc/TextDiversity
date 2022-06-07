import os
import sys

import pandas as pd
import numpy as np

def main():

   # google sheet that stores all the phonemic features for comparison
   url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRF7qR-gWbNCVmHYTW-UcXV6_O_eCh-EnCdGnDjHUdlEJkHyD_S25wh5pFUCfrtxbsauehkBQwad6i_/pub?gid=1821698515&single=true&output=csv"
   df = pd.read_csv(url, index_col=0)

   # categorical columns to turn into dummy variables
   cat_cols = [
       'Articulator Place', 
       'Articulator Intensity',
       'Articulator Position',
       'Articulator Class',
       'Active Articulator',
       'Passive Articulator',
       'Manner of Articulation',
       'Vowel Height',
       'Vowel Frontness',
       'Vowel Rounded'
   ]

   # final cols + the dummified cat_cols
   final_cols = [
       'IPA Symbol',
       'Consonant',
       'Voiced', 
       'Occlusive', 
       'Sonorant', 
       'Sibilant',
       'Strident', 
       'Semivowel', 
       'Vocoid', 
       'Continuant', 
       'Liquid', 
       'Rhotic',
       'Lateral', 
       'Vibrant'
   ]

   new_df = pd.concat([df[final_cols], pd.get_dummies(df[cat_cols], dummy_na=True, drop_first=True)], axis=1)
   new_df = new_df.set_index("IPA Symbol")

   n = len(new_df)
   store = np.zeros((n, n))
   for i in range(len(new_df)):
      for j in range(len(new_df)):
         sum_ = (new_df.iloc[i] == new_df.iloc[j]).sum()
         store[i, j] = sum_

   Z = pd.DataFrame(store, index=new_df.index.tolist(), columns=new_df.index.tolist())
   Z /= 70                    # divide by maximum possible overlap to normalize between 0 and 1
   Z **= 20                   # strongly penalize for being dissimilar (manually selected based on personally percieved similarity)
   Z = Z.where(Z > 0.1, 0)    # low-pass filter
   Z = Z.round(4)             # compress matrix

   print('saving phoneme_similarities.csv')
   script_dir = os.path.abspath(os.path.dirname(sys.argv[0]) or '.')
   csv_path = os.path.join(script_dir, '../phoneme_similarities.csv')
   Z.to_csv(csv_path)

if __name__ == '__main__':
   main()