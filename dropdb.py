import psycopg2

DB_SETTINGS = {
    'database': 'football_stats',
    'user': 'postgres',
    'password': 'Goyda_man1',
    'host': 'localhost',
    'port': '5432'
}

def drop_and_recreate_database():
    try:
        conn = psycopg2.connect(
            host=DB_SETTINGS['host'],
            user=DB_SETTINGS['user'],
            password=DB_SETTINGS['password'],
            port=DB_SETTINGS['port']
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        cursor.execute(f"""
            SELECT pg_terminate_backend(pg_stat_activity.pid)
            FROM pg_stat_activity
            WHERE pg_stat_activity.datname = '{DB_SETTINGS['database']}'
            AND pid <> pg_backend_pid();
        """)
        
        cursor.execute(f"DROP DATABASE IF EXISTS {DB_SETTINGS['database']}")
        print("База данных удалена")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Ошибка: {e}")
        return False

print("Сброс базы данных...")
drop_and_recreate_database()