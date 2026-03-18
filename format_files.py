import pandas as pd

df = pd.read_csv('data/la_liga25-26_last.csv')

df = df[['Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
        'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF',
        'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A', 'B365>2.5', 'B365<2.5']]

df.to_csv('data/la_liga25-26_last_cleaned.csv', index=False)