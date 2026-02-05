import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import os
import glob
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

def create_connection():
    try:
        conn = psycopg2.connect(**DB_SETTINGS)
        print("Успешное подключение")
        return conn
    except Exception as e:
        print(f"Ошибка подключения: {e}")
        return None

def create_database():
    try:
        conn = psycopg2.connect(
            host=DB_SETTINGS['host'],
            user=DB_SETTINGS['user'],
            password=DB_SETTINGS['password'],
            port=DB_SETTINGS['port']
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Проверка существования базы
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_SETTINGS['database']}'")
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(f"CREATE DATABASE {DB_SETTINGS['database']}")
            print("База данных создана")
        else:
            print("База данных уже существует")
            
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Ошибка при создании базы: {e}")
        return False

def create_tables():
    conn = create_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        # 1. Таблица с лигами
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS leagues (
                league_id SERIAL PRIMARY KEY,
                league_code VARCHAR(10) UNIQUE NOT NULL,
                league_name VARCHAR(100),
                country VARCHAR(50)
            )
        """)
        # 2. Таблица команд
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                team_id SERIAL PRIMARY KEY,
                team_name VARCHAR(100) UNIQUE NOT NULL,
                country VARCHAR(50)
            )
        """)
        # 3. Таблица судей
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS referees (
                referee_id SERIAL PRIMARY KEY,
                referee_name VARCHAR(100) UNIQUE NOT NULL
            )
        """)
        # 4. Основная таблица матчей
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                match_id SERIAL PRIMARY KEY,
                league_id INTEGER REFERENCES leagues(league_id),
                match_date DATE NOT NULL,
                match_time TIME,
                home_team_id INTEGER REFERENCES teams(team_id),
                away_team_id INTEGER REFERENCES teams(team_id),
                fthg INTEGER,
                ftag INTEGER,
                ftr CHAR(1),
                hthg INTEGER,
                htag INTEGER,
                htr CHAR(1),
                season VARCHAR(9),
                referee_id INTEGER REFERENCES referees(referee_id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # 5. Таблица статистики матча
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS match_stats (
                stat_id SERIAL PRIMARY KEY,
                match_id INTEGER REFERENCES matches(match_id) ON DELETE CASCADE,
                hs INTEGER,
                away_shots INTEGER,
                hst INTEGER,
                ast INTEGER,
                hc INTEGER,
                ac INTEGER,
                hf INTEGER,
                af INTEGER,
                hy INTEGER,
                ay INTEGER,
                hr INTEGER,
                ar INTEGER,
                UNIQUE(match_id)
            )
        """)
        # 6. Таблица коэффициентов букмекеров
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS odds (
                odds_id SERIAL PRIMARY KEY,
                match_id INTEGER REFERENCES matches(match_id) ON DELETE CASCADE,
                bookmaker VARCHAR(50) DEFAULT 'Bet365',
                home_win_odds DECIMAL(8,3),
                draw_odds DECIMAL(8,3),
                away_win_odds DECIMAL(8,3),
                over_2_5_odds DECIMAL(8,3),
                under_2_5_odds DECIMAL(8,3),
                odds_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Индексы для ускорения запросов
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_home_team ON matches(home_team_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_away_team ON matches(away_team_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(season)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_match ON match_stats(match_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_odds_match ON odds(match_id)")
        
        conn.commit()
        print("Таблицы созданы")
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Ошибка при создании таблиц: {e}")
        conn.close()
        return False

def clean_column_names(df):
    df = df.copy()
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('.', '_')
    df.columns = df.columns.str.replace('>', '_over_')
    df.columns = df.columns.str.replace('<', '_under_')
    # исправление as чтобы не читалось как команда
    if 'as' in df.columns:
        df = df.rename(columns={'as': 'away_shots'})
    
    # rename некоторых колонок
    rename_map = {
        'div': 'league_code',
        'date': 'match_date',
        'time': 'match_time',
        'hometeam': 'home_team',
        'awayteam': 'away_team'
    }
    
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    
    return df

def get_season_from_date(date_str):
    #определение сезона по дате
    try:
        if pd.isna(date_str):
            return None

        for fmt in ['%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d']:
            try:
                date = pd.to_datetime(date_str, format=fmt)
                year = date.year
                month = date.month
                
                # Если матч во второй половине года, это начало сезона
                if month >= 7:
                    return f"{year}-{year+1}"
                else:
                    return f"{year-1}-{year}"
                    
            except:
                continue
        return None
    except:
        return None

def insert_or_get_id(cursor, table, column, value, return_id_column='id'):
    if pd.isna(value):
        return None
    
    try:
        # Проверка существования
        query = f"SELECT {return_id_column} FROM {table} WHERE {column} = %s"
        cursor.execute(query, (value,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        else:
            # Вставка значения
            insert_query = f"INSERT INTO {table} ({column}) VALUES (%s) RETURNING {return_id_column}"
            cursor.execute(insert_query, (value,))
            return cursor.fetchone()[0]
    except Exception as e:
        print(f"Ошибка при работе с таблицей {table}: {e}")
        return None

def import_csv_files():
    #перенос данных в таблицы
    conn = create_connection()
    if not conn:
        return
    
    csv_files = glob.glob(os.path.join(DATA_FOLDER, '*cleaned.csv'))
    
    if not csv_files:
        print(f"Не найдено CSV файлов в папке '{DATA_FOLDER}'")
        return
    
    total_matches = 0
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            # Очищаем имена колонок
            df = clean_column_names(df)
            cursor = conn.cursor()
            for _, row in df.iterrows():
                # 1. ID лиги
                league_code = row.get('league_code', 'E0')
                league_id = insert_or_get_id(cursor, 'leagues', 'league_code', league_code, 'league_id')
                
                # 2. ID команд
                home_team_id = insert_or_get_id(cursor, 'teams', 'team_name', row.get('home_team'), 'team_id')
                away_team_id = insert_or_get_id(cursor, 'teams', 'team_name', row.get('away_team'), 'team_id')
                
                # 3. ID судьи
                referee_id = insert_or_get_id(cursor, 'referees', 'referee_name', row.get('referee'), 'referee_id')
                
                # 4. сезон
                match_date = pd.to_datetime(row.get('match_date'), dayfirst=True, errors='coerce')
                season = get_season_from_date(str(match_date))
                
                # 5. матч
                cursor.execute("""
                    INSERT INTO matches 
                    (league_id, match_date, match_time, home_team_id, away_team_id,
                     fthg, ftag, ftr, hthg, htag, htr, season, referee_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING match_id
                """, (
                    league_id, 
                    match_date, 
                    row.get('match_time'),
                    home_team_id, 
                    away_team_id,
                    row.get('fthg'), 
                    row.get('ftag'), 
                    row.get('ftr'),
                    row.get('hthg'), 
                    row.get('htag'), 
                    row.get('htr'),
                    season,
                    referee_id
                ))
                
                match_id = cursor.fetchone()[0]
                
                # 6. статистика
                cursor.execute("""
                    INSERT INTO match_stats 
                    (match_id, hs, away_shots, hst, ast, hc, ac, hf, af, hy, ay, hr, ar)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    match_id,
                    row.get('hs'),
                    row.get('away_shots'),
                    row.get('hst'),
                    row.get('ast'),
                    row.get('hc'),
                    row.get('ac'),
                    row.get('hf'),
                    row.get('af'),
                    row.get('hy'),
                    row.get('ay'),
                    row.get('hr'),
                    row.get('ar')
                ))
                
                # 7. коэффициенты
                cursor.execute("""
                    INSERT INTO odds 
                    (match_id, home_win_odds, draw_odds, away_win_odds, 
                     over_2_5_odds, under_2_5_odds)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    match_id,
                    row.get('b365h'),
                    row.get('b365d'),
                    row.get('b365a'),
                    row.get('b365_over_2_5'),
                    row.get('b365_under_2_5')
                ))
            
            conn.commit()
            total_matches += len(df)
            print(f"Импортировано матчей: {len(df)}")
            cursor.close()
            
        except Exception as e:
            print(f"Ошибка при обработке файла {filename}: {e}")
            conn.rollback()
            continue
    
    conn.close()
    print(f"данные загружены")
    
    return total_matches

def populate_league_info():
    #лиги
    conn = create_connection()
    if not conn:
        return
    try:
        cursor = conn.cursor()
        # Маппинг кодов лиг на названия и страны
        league_info = {
            'E0': ('Premier League', 'England'),
            'E1': ('Championship', 'England'),
            'E2': ('League One', 'England'),
            'E3': ('League Two', 'England'),
            'EC': ('Conference', 'England'),
            'D1': ('Bundesliga', 'Germany'),
            'D2': ('2. Bundesliga', 'Germany'),
            'I1': ('Serie A', 'Italy'),
            'I2': ('Serie B', 'Italy'),
            'SP1': ('La Liga', 'Spain'),
            'SP2': ('La Liga 2', 'Spain'),
            'F1': ('Ligue 1', 'France'),
            'F2': ('Ligue 2', 'France'),
            'N1': ('Eredivisie', 'Netherlands'),
            'P1': ('Primeira Liga', 'Portugal'),
        }
        
        for code, (name, country) in league_info.items():
            cursor.execute("""
                INSERT INTO leagues (league_code, league_name, country)
                VALUES (%s, %s, %s)
                ON CONFLICT (league_code) DO UPDATE SET
                league_name = EXCLUDED.league_name,
                country = EXCLUDED.country
            """, (code, name, country))
        
        conn.commit()
        print("Информация о лигах заполнена")
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Ошибка при заполнении информации о лигах: {e}")
        conn.close()


if not os.path.exists(DATA_FOLDER):
    print(f"Папка '{DATA_FOLDER}' не найдена!")
else:
    if not create_database():
        print("Не удалось создать базу данных")
    else:
        if not create_tables():
            print("Не удалось создать таблицы")
        else:
            populate_league_info()
            import_csv_files()
            
            print("\n" + "=" * 60)
            print("Готово! База данных успешно создана и заполнена.")
            print(f"   База данных: {DB_SETTINGS['database']}")
            print("   Созданные таблицы:")
            print("     1. leagues - Лиги")
            print("     2. teams - Команды")
            print("     3. referees - Судьи")
            print("     4. matches - Матчи")
            print("     5. match_stats - Статистика матчей")
            print("     6. odds - Коэффициенты букмекеров")
            print("=" * 60)