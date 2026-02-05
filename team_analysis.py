import pandas as pd
import numpy as np
import psycopg2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.WARNING)

DB_SETTINGS = {
    'database': 'football_stats',
    'user': 'postgres',
    'password': 'Goyda_man1',
    'host': 'localhost',
    'port': '5432'
}

class TeamStrengthAnalyzer:
    def __init__(self, min_matches=3, recent_matches=7):
        self.min_matches = min_matches
        self.recent_matches = recent_matches
        self.team_stats = {}
        self.team_weights = {}
        self.clusters = {}
        
    def connect(self):
        try:
            conn = psycopg2.connect(**DB_SETTINGS)
            return conn
        except Exception as e:
            logging.error(f"Ошибка подключения: {e}")
            return None
    
    def calculate_all_team_stats(self):
        conn = self.connect()
        if not conn:
            return None
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT team_id, team_name FROM teams")
            teams = cursor.fetchall()
            
            total_teams = len(teams)
            print(f"Найдено команд: {total_teams}")
            
            teams_excluded = 0
            teams_included = 0
            
            for idx, (team_id, team_name) in enumerate(teams, 1):
                stats = self.calculate_single_team_stats(cursor, team_id, team_name)
                
                # ОБНОВЛЕННАЯ ЛОГИКА: учитываем общее количество матчей
                total_matches = stats['home_total'] + stats['away_total']
                
                if total_matches >= self.min_matches:
                    self.team_stats[team_id] = stats
                    teams_included += 1
                else:
                    teams_excluded += 1
                    print(f"Команда {team_name} исключена: всего {total_matches} матчей (требуется {self.min_matches})")
            
            print(f"Рассчитана статистика для {teams_included} команд")
            print(f"Исключено {teams_excluded} команд из-за недостатка матчей")
            
            return self.team_stats
            
        except Exception as e:
            logging.error(f"Ошибка при расчете статистики: {e}")
            return None
        finally:
            cursor.close()
            conn.close()
    
    def calculate_single_team_stats(self, cursor, team_id, team_name):
        #Рассчитывает статистику для одной команды (вся доступная история)

        # 1. Домашняя статистика (вся история)
        home_query = """
        SELECT 
            COUNT(*) as total_matches,
            COUNT(CASE WHEN m.ftr = 'H' THEN 1 END) as wins,
            COUNT(CASE WHEN m.ftr = 'D' THEN 1 END) as draws,
            COUNT(CASE WHEN m.ftr = 'A' THEN 1 END) as losses,
            AVG(m.fthg) as avg_goals_scored,
            AVG(m.ftag) as avg_goals_conceded,
            COUNT(CASE WHEN m.ftag = 0 THEN 1 END) as clean_sheets,
            COUNT(CASE WHEN m.ftr IN ('H', 'D') THEN 1 END) as unbeaten
        FROM matches m
        WHERE m.home_team_id = %s
        """

        cursor.execute(home_query, [team_id])
        result = cursor.fetchone()

        if result and result[0] is not None:
            # Преобразуем ВСЕ значения в float при распаковке
            home_total = float(result[0]) if result[0] is not None else 0.0
            home_wins = float(result[1]) if result[1] is not None else 0.0
            home_draws = float(result[2]) if result[2] is not None else 0.0
            home_losses = float(result[3]) if result[3] is not None else 0.0
            home_avg_scored = float(result[4]) if result[4] is not None else 0.0
            home_avg_conceded = float(result[5]) if result[5] is not None else 0.0
            home_clean_sheets = float(result[6]) if result[6] is not None else 0.0
            home_unbeaten = float(result[7]) if result[7] is not None else 0.0
        else:
            home_total = home_wins = home_draws = home_losses = 0.0
            home_avg_scored = home_avg_conceded = 0.0
            home_clean_sheets = home_unbeaten = 0.0

        # 2. Выездная статистика (вся история)
        away_query = """
        SELECT 
            COUNT(*) as total_matches,
            COUNT(CASE WHEN m.ftr = 'A' THEN 1 END) as wins,
            COUNT(CASE WHEN m.ftr = 'D' THEN 1 END) as draws,
            COUNT(CASE WHEN m.ftr = 'H' THEN 1 END) as losses,
            AVG(m.ftag) as avg_goals_scored,
            AVG(m.fthg) as avg_goals_conceded,
            COUNT(CASE WHEN m.fthg = 0 THEN 1 END) as clean_sheets,
            COUNT(CASE WHEN m.ftr IN ('A', 'D') THEN 1 END) as unbeaten
        FROM matches m
        WHERE m.away_team_id = %s
        """

        cursor.execute(away_query, [team_id])
        result = cursor.fetchone()

        if result and result[0] is not None:
            # Преобразуем ВСЕ значения в float при распаковке
            away_total = float(result[0]) if result[0] is not None else 0.0
            away_wins = float(result[1]) if result[1] is not None else 0.0
            away_draws = float(result[2]) if result[2] is not None else 0.0
            away_losses = float(result[3]) if result[3] is not None else 0.0
            away_avg_scored = float(result[4]) if result[4] is not None else 0.0
            away_avg_conceded = float(result[5]) if result[5] is not None else 0.0
            away_clean_sheets = float(result[6]) if result[6] is not None else 0.0
            away_unbeaten = float(result[7]) if result[7] is not None else 0.0
        else:
            away_total = away_wins = away_draws = away_losses = 0.0
            away_avg_scored = away_avg_conceded = 0.0
            away_clean_sheets = away_unbeaten = 0.0

        # 3. Форма в последних матчах (до 7 матчей, но не больше доступных)
        # Сначала подсчитаем, сколько матчей доступно
        available_matches_query = """
        SELECT COUNT(*)
        FROM matches m
        WHERE (m.home_team_id = %s OR m.away_team_id = %s)
        """

        cursor.execute(available_matches_query, [team_id, team_id])
        result = cursor.fetchone()
        available_matches_count = float(result[0]) if result and result[0] is not None else 0.0

        # Определяем, сколько матчей использовать для формы
        matches_for_form = min(available_matches_count, float(self.recent_matches))

        # Если есть хотя бы один матч для анализа формы
        if matches_for_form > 0:
            form_query = """
            WITH recent_matches AS (
                SELECT 
                    m.match_date,
                    CASE 
                        WHEN m.home_team_id = %s THEN 
                            CASE m.ftr
                                WHEN 'H' THEN 3
                                WHEN 'D' THEN 1
                                ELSE 0
                            END
                        ELSE
                            CASE m.ftr
                                WHEN 'A' THEN 3
                                WHEN 'D' THEN 1
                                ELSE 0
                            END
                    END as points,
                    CASE 
                        WHEN m.home_team_id = %s THEN m.fthg
                        ELSE m.ftag
                    END as goals_scored,
                    CASE 
                        WHEN m.home_team_id = %s THEN m.ftag
                        ELSE m.fthg
                    END as goals_conceded
                FROM matches m
                WHERE (m.home_team_id = %s OR m.away_team_id = %s)
                ORDER BY m.match_date DESC
                LIMIT %s
            )
            SELECT 
                COUNT(*) as matches_count,
                COALESCE(SUM(points), 0) as total_points,
                COALESCE(AVG(points), 0) as avg_points,
                COALESCE(SUM(goals_scored), 0) as total_goals_scored,
                COALESCE(SUM(goals_conceded), 0) as total_goals_conceded,
                COALESCE(AVG(goals_scored), 0) as avg_goals_scored,
                COALESCE(AVG(goals_conceded), 0) as avg_goals_conceded
            FROM recent_matches
            """
            
            form_params = [team_id, team_id, team_id, team_id, team_id, int(matches_for_form)]
            cursor.execute(form_query, form_params)
            result = cursor.fetchone()
            
            if result:
                # Преобразуем ВСЕ значения в float при распаковке
                form_matches = float(result[0]) if result[0] is not None else 0.0
                form_points = float(result[1]) if result[1] is not None else 0.0
                form_avg_points = float(result[2]) if result[2] is not None else 0.0
                form_goals_scored = float(result[3]) if result[3] is not None else 0.0
                form_goals_conceded = float(result[4]) if result[4] is not None else 0.0
                form_avg_scored = float(result[5]) if result[5] is not None else 0.0
                form_avg_conceded = float(result[6]) if result[6] is not None else 0.0
            else:
                form_matches = form_points = form_avg_points = 0.0
                form_goals_scored = form_goals_conceded = form_avg_scored = form_avg_conceded = 0.0
        else:
            form_matches = form_points = form_avg_points = 0.0
            form_goals_scored = form_goals_conceded = form_avg_scored = form_avg_conceded = 0.0

        # 4. Текущая серия без поражений дома (последние 10 матчей)
        home_streak_query = """
        WITH home_matches AS (
            SELECT 
                m.match_date,
                m.ftr
            FROM matches m
            WHERE m.home_team_id = %s
            ORDER BY m.match_date DESC
            LIMIT 10
        )
        SELECT COUNT(*) as streak
        FROM (
            SELECT ftr,
                ROW_NUMBER() OVER (ORDER BY match_date DESC) as rn
            FROM home_matches
        ) t
        WHERE t.ftr != 'A'
        """

        cursor.execute(home_streak_query, [team_id])
        result = cursor.fetchone()
        home_unbeaten_streak = float(result[0]) if result and result[0] is not None else 0.0

        # 5. Текущая серия без поражений в гостях (последние 10 матчей)
        away_streak_query = """
        WITH away_matches AS (
            SELECT 
                m.match_date,
                m.ftr
            FROM matches m
            WHERE m.away_team_id = %s
            ORDER BY m.match_date DESC
            LIMIT 10
        )
        SELECT COUNT(*) as streak
        FROM (
            SELECT ftr,
                ROW_NUMBER() OVER (ORDER BY match_date DESC) as rn
            FROM away_matches
        ) t
        WHERE t.ftr != 'H'
        """

        cursor.execute(away_streak_query, [team_id])
        result = cursor.fetchone()
        away_unbeaten_streak = float(result[0]) if result and result[0] is not None else 0.0

        # Расчет процентов и рейтингов (теперь все значения уже float)
        home_win_rate = home_wins / home_total if home_total > 0 else 0.0
        away_win_rate = away_wins / away_total if away_total > 0 else 0.0

        home_clean_sheets_rate = home_clean_sheets / home_total if home_total > 0 else 0.0
        away_clean_sheets_rate = away_clean_sheets / away_total if away_total > 0 else 0.0

        home_unbeaten_rate = home_unbeaten / home_total if home_total > 0 else 0.0
        away_unbeaten_rate = away_unbeaten / away_total if away_total > 0 else 0.0

        form_rating = form_avg_points / 3.0 if form_matches > 0 else 0.5

        # Формируем результат (все значения уже float)
        stats = {
            'team_id': team_id,  # оставляем как есть (int)
            'team_name': team_name,  # оставляем как есть (str)
            
            # Домашние показатели
            'home_total': home_total,
            'home_win_rate': home_win_rate,
            'home_clean_sheets_rate': home_clean_sheets_rate,
            'home_unbeaten_rate': home_unbeaten_rate,
            'home_avg_scored': home_avg_scored,
            'home_avg_conceded': home_avg_conceded,
            'home_unbeaten_streak': home_unbeaten_streak,
            
            # Выездные показатели
            'away_total': away_total,
            'away_win_rate': away_win_rate,
            'away_clean_sheets_rate': away_clean_sheets_rate,
            'away_unbeaten_rate': away_unbeaten_rate,
            'away_avg_scored': away_avg_scored,
            'away_avg_conceded': away_avg_conceded,
            'away_unbeaten_streak': away_unbeaten_streak,
            
            # Форма
            'form_matches': form_matches,
            'form_points': form_points,
            'form_avg_points': form_avg_points,
            'form_rating': form_rating,
            'form_goals_scored': form_goals_scored,
            'form_goals_conceded': form_goals_conceded,
            'form_avg_scored': form_avg_scored,
            'form_avg_conceded': form_avg_conceded,
            
            # Сводные показатели
            'overall_win_rate': (home_wins + away_wins) / (home_total + away_total) if (home_total + away_total) > 0 else 0.0,
            'points_per_game': ((home_wins * 3.0 + home_draws) + (away_wins * 3.0 + away_draws)) / (home_total + away_total) if (home_total + away_total) > 0 else 0.0,
        }

        return stats
    
    def cluster_teams_by_strength(self, n_clusters):
        #Кластеризует команды по силе
        print(f"Кластеризация команд на {n_clusters} кластера...")
        
        if not self.team_stats:
            logging.error("Сначала нужно рассчитать статистику команд")
            return None
        
        # Подготавливаем данные для кластеризации
        data = []
        team_ids = []
        
        for team_id, stats in self.team_stats.items():
            team_ids.append(team_id)
            data.append([
                stats['home_win_rate'],
                stats['away_win_rate'],
                stats['home_clean_sheets_rate'],
                stats['away_clean_sheets_rate'],
                stats['form_rating'],
                stats['overall_win_rate'],
                stats['points_per_game']
            ])
        
        # Преобразуем в numpy array
        X = np.array(data)
        
        # Масштабируем данные
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Кластеризация K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Сохраняем результаты
        self.clusters = dict(zip(team_ids, clusters))
        
        # Анализируем кластеры
        self.analyze_clusters()
        
        return self.clusters
    
    def analyze_clusters(self):
        #Анализирует и описывает кластеры
        if not self.clusters:
            return
        
        # Группируем команды по кластерам
        cluster_teams = {}
        for team_id, cluster in self.clusters.items():
            if cluster not in cluster_teams:
                cluster_teams[cluster] = []
            cluster_teams[cluster].append(self.team_stats[team_id]['team_name'])
        
        # Анализируем статистику по кластерам
        cluster_stats = {}
        
        for cluster_id in sorted(cluster_teams.keys()):
            teams_in_cluster = [tid for tid, c in self.clusters.items() if c == cluster_id]
            
            # Собираем статистику кластера
            stats_list = [self.team_stats[tid] for tid in teams_in_cluster]
            
            cluster_stats[cluster_id] = {
                'team_count': len(teams_in_cluster),
                'teams': cluster_teams[cluster_id],
                'avg_home_win_rate': np.mean([s['home_win_rate'] for s in stats_list]),
                'avg_away_win_rate': np.mean([s['away_win_rate'] for s in stats_list]),
                'avg_form_rating': np.mean([s['form_rating'] for s in stats_list]),
                'avg_points_per_game': np.mean([s['points_per_game'] for s in stats_list])
            }
        
        # Выводим анализ
        for cluster_id, stats in sorted(cluster_stats.items()):
            print(f"\nКластер {cluster_id} ({stats['team_count']} команд):")
            print(f"  Домашние победы: {stats['avg_home_win_rate']:.1%}")
            print(f"  Выездные победы: {stats['avg_away_win_rate']:.1%}")
            print(f"  Форма: {stats['avg_form_rating']:.2f}")
            print(f"  Очки за матч: {stats['avg_points_per_game']:.2f}")
            print(f"  Примеры команд: {', '.join(stats['teams'][:5])}")
        
        return cluster_stats
    
    def calculate_team_weights(self):
        #Рассчитывает веса для команд на основе их силы
        
        if not self.team_stats:
            logging.error("Сначала нужно рассчитать статистику команд")
            return None
        
        self.team_weights = {}
        
        for team_id, stats in self.team_stats.items():
            # Базовые веса от 0 до 1
            weights = {
                'home_win_rate': stats['home_win_rate'],
                'away_win_rate': stats['away_win_rate'],
                'home_clean_sheets_rate': stats['home_clean_sheets_rate'],
                'away_clean_sheets_rate': stats['away_clean_sheets_rate'],
                'form_rating': stats['form_rating'],
                'home_unbeaten_streak': min(stats['home_unbeaten_streak'] / 10, 1.0),  # Нормализуем к 0-1
                'away_unbeaten_streak': min(stats['away_unbeaten_streak'] / 10, 1.0),  # Нормализуем к 0-1
                'overall_strength': stats['overall_win_rate'],
                'cluster': self.clusters.get(team_id, 0) / 3 if self.clusters else 0.5,  # Нормализуем кластер к 0-1
            }
            
            # Добавляем составной вес (среднее всех показателей)
            weights['composite_weight'] = np.mean([
                weights['home_win_rate'],
                weights['away_win_rate'],
                weights['form_rating'],
                weights['overall_strength']
            ])
            
            self.team_weights[team_id] = weights
        
        # Нормализуем составные веса к диапазону 0.3-1.0
        composite_weights = [w['composite_weight'] for w in self.team_weights.values()]
        min_weight, max_weight = min(composite_weights), max(composite_weights)
        
        for team_id in self.team_weights:
            if max_weight > min_weight:
                # Нормализуем к 0.3-1.0
                normalized = 0.3 + 0.7 * (self.team_weights[team_id]['composite_weight'] - min_weight) / (max_weight - min_weight)
                self.team_weights[team_id]['normalized_weight'] = normalized
            else:
                self.team_weights[team_id]['normalized_weight'] = 0.65
        
        print(f"Рассчитаны веса для {len(self.team_weights)} команд")
        
        return self.team_weights
    
    def get_team_features_for_match(self, home_team_id, away_team_id):
        #Возвращает признаки для конкретного матча
        home_stats = self.team_stats.get(home_team_id)
        away_stats = self.team_stats.get(away_team_id)
        
        if not home_stats or not away_stats:
            return None
        
        home_weights = self.team_weights.get(home_team_id, {})
        away_weights = self.team_weights.get(away_team_id, {})
        
        # Базовые признаки
        features = {
            # Домашняя команда
            'home_win_rate': home_stats['home_win_rate'],
            'home_clean_sheets_rate': home_stats['home_clean_sheets_rate'],
            'home_unbeaten_streak': home_stats['home_unbeaten_streak'],
            'home_form_rating': home_stats['form_rating'],
            
            # Гостевая команда
            'away_win_rate': away_stats['away_win_rate'],
            'away_clean_sheets_rate': away_stats['away_clean_sheets_rate'],
            'away_unbeaten_streak': away_stats['away_unbeaten_streak'],
            'away_form_rating': away_stats['form_rating'],
            
            # Разницы
            'win_rate_difference': home_stats['home_win_rate'] - away_stats['away_win_rate'],
            'form_difference': home_stats['form_rating'] - away_stats['form_rating'],
            'clean_sheets_difference': home_stats['home_clean_sheets_rate'] - away_stats['away_clean_sheets_rate'],
            
            # Веса
            'home_weight': home_weights.get('normalized_weight', 0.5),
            'away_weight': away_weights.get('normalized_weight', 0.5),
            'weight_difference': home_weights.get('normalized_weight', 0.5) - away_weights.get('normalized_weight', 0.5),
            
            # Кластеры
            'home_cluster': self.clusters.get(home_team_id, 0),
            'away_cluster': self.clusters.get(away_team_id, 0),
            'cluster_difference': self.clusters.get(home_team_id, 0) - self.clusters.get(away_team_id, 0),
        }
        
        return features
    
    def save_results(self, filename='team_strength_analysis.json'):
        #Сохраняет результаты анализа в JSON файл
        results = {
            'analysis_date': datetime.now().isoformat(),
            'min_matches': self.min_matches,
            'recent_matches': self.recent_matches,
            'team_stats': self.team_stats,
            'team_weights': self.team_weights,
            'clusters': self.clusters
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        return filename



if __name__ == '__main__':
    # Инициализация анализатора
    analyzer = TeamStrengthAnalyzer(
        min_matches=3,      # Минимум 3 матча дома/в гостях
        recent_matches=7    # Форма по последним 7 матчам
    )
    
    stats = analyzer.calculate_all_team_stats()
    
    if not stats:
        print("Не удалось рассчитать статистику")
    
    # 2. Кластеризация команд
    clusters = analyzer.cluster_teams_by_strength(n_clusters=4)
    
    # 3. Расчет весов команд
    weights = analyzer.calculate_team_weights()
    
    # 6. Сохранение результатов
    analyzer.save_results('team_strength_results.json')