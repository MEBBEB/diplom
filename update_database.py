"""
update_database.py
Скрипт для добавления новых матчей из CSV файлов в БД
"""

import pandas as pd
import psycopg2
import glob
import os
from datetime import datetime

# настройки бд
DB_SETTINGS = {
    'database': 'football_stats',
    'user': 'postgres',
    'password': 'Goyda_man1',
    'host': 'localhost',
    'port': '5432'
}

DATA_FOLDER = 'data'

def clean_column_names(df):
    """Очистка имен колонок"""
    df = df.copy()
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('.', '_')
    if 'as' in df.columns:
        df = df.rename(columns={'as': 'away_shots'})
    rename_map = {
        'div': 'league_code',
        'date': 'match_date',
        'time': 'match_time',
        'hometeam': 'home_team',
        'awayteam': 'away_team'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df

def get_season_from_date(date_str):
    """Определение сезона по дате"""
    try:
        if pd.isna(date_str):
            return None
        date = pd.to_datetime(date_str, dayfirst=True)
        year = date.year
        month = date.month
        return f"{year}-{year+1}" if month >= 7 else f"{year-1}-{year}"
    except:
        return None

def insert_or_get_id(cursor, table, column, value, id_column='id'):
    """Получение ID существующей записи или вставка новой"""
    if pd.isna(value):
        return None
    cursor.execute(f"SELECT {id_column} FROM {table} WHERE {column} = %s", (value,))
    result = cursor.fetchone()
    if result:
        return result[0]
    cursor.execute(f"INSERT INTO {table} ({column}) VALUES (%s) RETURNING {id_column}", (value,))
    return cursor.fetchone()[0]

def match_exists(cursor, match_date, home_id, away_id):
    """Проверка существования матча"""
    cursor.execute("""
        SELECT match_id FROM matches 
        WHERE match_date = %s AND home_team_id = %s AND away_team_id = %s
    """, (match_date, home_id, away_id))
    return cursor.fetchone() is not None

def update_database():
    """Основная функция обновления БД"""
    print(f"\n{'='*60}")
    print(f"ОБНОВЛЕНИЕ БАЗЫ ДАННЫХ ИЗ {DATA_FOLDER}")
    print(f"{'='*60}")
    
    # Поиск CSV файлов
    csv_files = glob.glob(os.path.join(DATA_FOLDER, '*last_cleaned.csv'))
    if not csv_files:
        print(f"Файлы не найдены в папке {DATA_FOLDER}")
        return
    
    print(f"Найдено файлов: {len(csv_files)}")
    
    # Подключение к БД
    conn = psycopg2.connect(**DB_SETTINGS)
    cursor = conn.cursor()
    
    total_new = 0
    total_skipped = 0
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"\nОбработка {filename}...")
        
        # Чтение CSV
        df = pd.read_csv(file_path, encoding='utf-8')
        df = clean_column_names(df)
        print(f"  Всего записей: {len(df)}")
        
        file_new = 0
        file_skipped = 0
        
        for _, row in df.iterrows():
            try:
                # Получаем ID лиги
                league_code = row.get('league_code', 'E0')
                league_id = insert_or_get_id(cursor, 'leagues', 'league_code', league_code, 'league_id')
                
                # Получаем ID команд
                home_team_id = insert_or_get_id(cursor, 'teams', 'team_name', row.get('home_team'), 'team_id')
                away_team_id = insert_or_get_id(cursor, 'teams', 'team_name', row.get('away_team'), 'team_id')
                
                if not home_team_id or not away_team_id:
                    file_skipped += 1
                    continue
                
                # Получаем ID судьи
                referee_id = insert_or_get_id(cursor, 'referees', 'referee_name', row.get('referee'), 'referee_id')
                
                # Обработка даты
                match_date = pd.to_datetime(row.get('match_date'), dayfirst=True).date()
                season = get_season_from_date(str(match_date))
                
                # Проверка существования матча
                if match_exists(cursor, match_date, home_team_id, away_team_id):
                    file_skipped += 1
                    continue
                
                # Вставка матча
                cursor.execute("""
                    INSERT INTO matches 
                    (league_id, match_date, match_time, home_team_id, away_team_id,
                     fthg, ftag, ftr, hthg, htag, htr, season, referee_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING match_id
                """, (
                    league_id, match_date, row.get('match_time'),
                    home_team_id, away_team_id,
                    row.get('fthg'), row.get('ftag'), row.get('ftr'),
                    row.get('hthg'), row.get('htag'), row.get('htr'),
                    season, referee_id
                ))
                
                match_id = cursor.fetchone()[0]
                
                # Вставка статистики
                cursor.execute("""
                    INSERT INTO match_stats 
                    (match_id, hs, away_shots, hst, ast, hc, ac, hf, af, hy, ay, hr, ar)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    match_id, row.get('hs'), row.get('away_shots'), row.get('hst'),
                    row.get('ast'), row.get('hc'), row.get('ac'), row.get('hf'),
                    row.get('af'), row.get('hy'), row.get('ay'), row.get('hr'), row.get('ar')
                ))
                
                # Вставка коэффициентов
                cursor.execute("""
                    INSERT INTO odds 
                    (match_id, home_win_odds, draw_odds, away_win_odds, 
                     over_2_5_odds, under_2_5_odds)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    match_id, row.get('b365h'), row.get('b365d'), row.get('b365a'),
                    row.get('b365_over_2_5'), row.get('b365_under_2_5')
                ))
                
                file_new += 1
                
            except Exception as e:
                print(f"    Ошибка: {e}")
                file_skipped += 1
                continue
        
        conn.commit()
        print(f"  Новых: {file_new}, Пропущено: {file_skipped}")
        total_new += file_new
        total_skipped += file_skipped
    
    cursor.close()
    conn.close()
    
    print(f"\n{'='*60}")
    print(f"ОБНОВЛЕНИЕ ЗАВЕРШЕНО")
    print(f"  Добавлено новых матчей: {total_new}")
    print(f"  Пропущено (уже есть): {total_skipped}")
    print(f"{'='*60}")

if __name__ == "__main__":
    update_database()