"""
team_analysis.py
Классический анализ силы команд с использованием статистики из БД
"""

import pandas as pd
import numpy as np
import psycopg2
import json
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_SETTINGS = {
    'database': 'football_stats',
    'user': 'postgres',
    'password': 'Goyda_man1',
    'host': 'localhost',
    'port': '5432'
}

class TeamStrengthAnalyzer:
    """
    Анализатор силы команд на основе исторической статистики
    """
    
    def __init__(self, min_matches=3, recent_matches=5):
        """
        Инициализация анализатора
        
        Args:
            min_matches: минимальное количество матчей для анализа команды
            recent_matches: количество последних матчей для оценки формы
        """
        self.min_matches = min_matches
        self.recent_matches = recent_matches
        self.team_stats = {}
        self.team_weights = {}
        
    def connect(self):
        """Подключение к базе данных"""
        try:
            conn = psycopg2.connect(**DB_SETTINGS)
            return conn
        except Exception as e:
            logger.error(f"Ошибка подключения: {e}")
            return None
    
    def calculate_all_team_stats(self):
        """
        Расчет статистики для всех команд
        """
        conn = self.connect()
        if not conn:
            return None
            
        try:
            cursor = conn.cursor()
            
            # Получаем все команды
            cursor.execute("""
                SELECT team_id, team_name 
                FROM teams 
                ORDER BY team_name
            """)
            teams = cursor.fetchall()
            
            total_teams = len(teams)
            logger.info(f"Найдено команд: {total_teams}")
            
            teams_excluded = 0
            teams_included = 0
            
            for idx, (team_id, team_name) in enumerate(teams, 1):
                stats = self._calculate_single_team_stats(cursor, team_id, team_name)
                
                # Проверяем минимальное количество матчей
                total_matches = stats['home_total'] + stats['away_total']
                
                if total_matches >= self.min_matches:
                    self.team_stats[team_id] = stats
                    teams_included += 1
                else:
                    teams_excluded += 1
                    logger.debug(f"Команда {team_name} исключена: всего {total_matches} матчей")
            
            logger.info(f"Рассчитана статистика для {teams_included} команд")
            logger.info(f"Исключено {teams_excluded} команд из-за недостатка матчей")
            
            return self.team_stats
            
        except Exception as e:
            logger.error(f"Ошибка при расчете статистики: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def _calculate_single_team_stats(self, cursor, team_id, team_name):
        """
        Расчет статистики для одной команды
        """
        # 1. Домашняя статистика
        home_query = """
        SELECT 
            COUNT(*) as total_matches,
            COUNT(CASE WHEN ftr = 'H' THEN 1 END) as wins,
            COUNT(CASE WHEN ftr = 'D' THEN 1 END) as draws,
            COUNT(CASE WHEN ftr = 'A' THEN 1 END) as losses,
            AVG(fthg) as avg_goals_scored,
            AVG(ftag) as avg_goals_conceded,
            SUM(fthg - ftag) as goal_difference,
            COUNT(CASE WHEN ftag = 0 THEN 1 END) as clean_sheets
        FROM matches
        WHERE home_team_id = %s
        """
        
        cursor.execute(home_query, [team_id])
        result = cursor.fetchone()
        
        # Обработка результатов с защитой от NULL
        home_total = float(result[0]) if result and result[0] is not None else 0.0
        home_wins = float(result[1]) if result and result[1] is not None else 0.0
        home_draws = float(result[2]) if result and result[2] is not None else 0.0
        home_losses = float(result[3]) if result and result[3] is not None else 0.0
        home_avg_scored = float(result[4]) if result and result[4] is not None else 0.0
        home_avg_conceded = float(result[5]) if result and result[5] is not None else 0.0
        home_goal_diff = float(result[6]) if result and result[6] is not None else 0.0
        home_clean_sheets = float(result[7]) if result and result[7] is not None else 0.0

        # 2. Выездная статистика
        away_query = """
        SELECT 
            COUNT(*) as total_matches,
            COUNT(CASE WHEN ftr = 'A' THEN 1 END) as wins,
            COUNT(CASE WHEN ftr = 'D' THEN 1 END) as draws,
            COUNT(CASE WHEN ftr = 'H' THEN 1 END) as losses,
            AVG(ftag) as avg_goals_scored,
            AVG(fthg) as avg_goals_conceded,
            SUM(ftag - fthg) as goal_difference,
            COUNT(CASE WHEN fthg = 0 THEN 1 END) as clean_sheets
        FROM matches
        WHERE away_team_id = %s
        """
        
        cursor.execute(away_query, [team_id])
        result = cursor.fetchone()
        
        away_total = float(result[0]) if result and result[0] is not None else 0.0
        away_wins = float(result[1]) if result and result[1] is not None else 0.0
        away_draws = float(result[2]) if result and result[2] is not None else 0.0
        away_losses = float(result[3]) if result and result[3] is not None else 0.0
        away_avg_scored = float(result[4]) if result and result[4] is not None else 0.0
        away_avg_conceded = float(result[5]) if result and result[5] is not None else 0.0
        away_goal_diff = float(result[6]) if result and result[6] is not None else 0.0
        away_clean_sheets = float(result[7]) if result and result[7] is not None else 0.0

        # 3. Текущая серия без поражений ДОМА
        home_streak_query = """
        WITH home_matches AS (
            SELECT 
                match_date,
                CASE WHEN ftr = 'A' THEN 1 ELSE 0 END as is_loss
            FROM matches
            WHERE home_team_id = %s
            ORDER BY match_date DESC
        )
        SELECT COUNT(*) as current_streak
        FROM home_matches
        WHERE match_date > COALESCE(
            (SELECT MAX(match_date) FROM home_matches WHERE is_loss = 1),
            '1900-01-01'::date
        )
        """
        
        cursor.execute(home_streak_query, [team_id])
        result = cursor.fetchone()
        home_unbeaten_streak = float(result[0]) if result and result[0] is not None else 0.0

        # 4. Текущая серия без поражений В ГОСТЯХ
        away_streak_query = """
        WITH away_matches AS (
            SELECT 
                match_date,
                CASE WHEN ftr = 'H' THEN 1 ELSE 0 END as is_loss
            FROM matches
            WHERE away_team_id = %s
            ORDER BY match_date DESC
        )
        SELECT COUNT(*) as current_streak
        FROM away_matches
        WHERE match_date > COALESCE(
            (SELECT MAX(match_date) FROM away_matches WHERE is_loss = 1),
            '1900-01-01'::date
        )
        """
        
        cursor.execute(away_streak_query, [team_id])
        result = cursor.fetchone()
        away_unbeaten_streak = float(result[0]) if result and result[0] is not None else 0.0

        # 5. Форма в последних N матчах
        form_query = """
        WITH recent_matches AS (
            SELECT 
                match_date,
                CASE 
                    WHEN home_team_id = %s THEN 
                        CASE ftr
                            WHEN 'H' THEN 3
                            WHEN 'D' THEN 1
                            ELSE 0
                        END
                    ELSE
                        CASE ftr
                            WHEN 'A' THEN 3
                            WHEN 'D' THEN 1
                            ELSE 0
                        END
                END as points,
                CASE 
                    WHEN home_team_id = %s THEN fthg
                    ELSE ftag
                END as goals_scored,
                CASE 
                    WHEN home_team_id = %s THEN ftag
                    ELSE fthg
                END as goals_conceded
            FROM matches
            WHERE (home_team_id = %s OR away_team_id = %s)
            ORDER BY match_date DESC
            LIMIT %s
        )
        SELECT 
            COUNT(*) as matches_count,
            COALESCE(SUM(points), 0) as total_points,
            COALESCE(AVG(points), 0) as avg_points,
            COALESCE(SUM(goals_scored), 0) as total_goals_scored,
            COALESCE(SUM(goals_conceded), 0) as total_goals_conceded,
            COALESCE(SUM(goals_scored - goals_conceded), 0) as form_goal_diff
        FROM recent_matches
        """
        
        form_params = [team_id, team_id, team_id, team_id, team_id, self.recent_matches]
        cursor.execute(form_query, form_params)
        result = cursor.fetchone()
        
        if result:
            form_matches = float(result[0]) if result[0] is not None else 0.0
            form_points = float(result[1]) if result[1] is not None else 0.0
            form_avg_points = float(result[2]) if result[2] is not None else 0.0
            form_goals_scored = float(result[3]) if result[3] is not None else 0.0
            form_goals_conceded = float(result[4]) if result[4] is not None else 0.0
            form_goal_diff = float(result[5]) if result[5] is not None else 0.0
        else:
            form_matches = form_points = form_avg_points = 0.0
            form_goals_scored = form_goals_conceded = form_goal_diff = 0.0

        # Расчет процентов
        home_win_rate = home_wins / home_total if home_total > 0 else 0.0
        away_win_rate = away_wins / away_total if away_total > 0 else 0.0
        home_clean_sheets_rate = home_clean_sheets / home_total if home_total > 0 else 0.0
        away_clean_sheets_rate = away_clean_sheets / away_total if away_total > 0 else 0.0
        
        form_rating = form_avg_points / 3.0 if form_matches > 0 else 0.5
        
        # Разница голов за матч
        home_goal_diff_per_game = home_goal_diff / home_total if home_total > 0 else 0.0
        away_goal_diff_per_game = away_goal_diff / away_total if away_total > 0 else 0.0
        
        total_matches = home_total + away_total
        total_goals_scored = (home_avg_scored * home_total) + (away_avg_scored * away_total)
        total_goals_conceded = (home_avg_conceded * home_total) + (away_avg_conceded * away_total)
        
        # Формируем результат
        stats = {
            'team_id': team_id,
            'team_name': team_name,
            
            # Домашние показатели
            'home_total': home_total,
            'home_wins': home_wins,
            'home_draws': home_draws,
            'home_losses': home_losses,
            'home_win_rate': home_win_rate,
            'home_clean_sheets_rate': home_clean_sheets_rate,
            'home_unbeaten_streak': home_unbeaten_streak,
            'home_avg_scored': home_avg_scored,
            'home_avg_conceded': home_avg_conceded,
            'home_goal_diff': home_goal_diff,
            'home_goal_diff_per_game': home_goal_diff_per_game,
            
            # Выездные показатели
            'away_total': away_total,
            'away_wins': away_wins,
            'away_draws': away_draws,
            'away_losses': away_losses,
            'away_win_rate': away_win_rate,
            'away_clean_sheets_rate': away_clean_sheets_rate,
            'away_unbeaten_streak': away_unbeaten_streak,
            'away_avg_scored': away_avg_scored,
            'away_avg_conceded': away_avg_conceded,
            'away_goal_diff': away_goal_diff,
            'away_goal_diff_per_game': away_goal_diff_per_game,
            
            # Форма
            'form_matches': form_matches,
            'form_points': form_points,
            'form_avg_points': form_avg_points,
            'form_rating': form_rating,
            'form_goals_scored': form_goals_scored,
            'form_goals_conceded': form_goals_conceded,
            'form_goal_diff': form_goal_diff,
            
            # Сводные показатели
            'total_matches': total_matches,
            'overall_win_rate': (home_wins + away_wins) / total_matches if total_matches > 0 else 0.0,
            'points_per_game': ((home_wins * 3 + home_draws) + (away_wins * 3 + away_draws)) / total_matches if total_matches > 0 else 0.0,
            'total_goal_diff': home_goal_diff + away_goal_diff,
            'goal_diff_per_game': (home_goal_diff + away_goal_diff) / total_matches if total_matches > 0 else 0.0,
            'avg_scored_per_game': total_goals_scored / total_matches if total_matches > 0 else 0.0,
            'avg_conceded_per_game': total_goals_conceded / total_matches if total_matches > 0 else 0.0,
        }

        return stats
    
    def calculate_team_weights(self):
        """
        Расчет весов для команд на основе их силы
        """
        if not self.team_stats:
            logger.error("Сначала нужно рассчитать статистику команд")
            return None
        
        self.team_weights = {}
        
        for team_id, stats in self.team_stats.items():
            # Нормализуем разницу голов
            goal_diff_score = min(max(stats['goal_diff_per_game'] / 2 + 0.5, 0), 1)
            form_goal_diff_score = min(max(stats['form_goal_diff'] / 10 + 0.5, 0), 1)
            
            weights = {
                'home_win_rate': stats['home_win_rate'],
                'away_win_rate': stats['away_win_rate'],
                'home_clean_sheets_rate': stats['home_clean_sheets_rate'],
                'away_clean_sheets_rate': stats['away_clean_sheets_rate'],
                'form_rating': stats['form_rating'],
                'goal_diff_score': goal_diff_score,
                'form_goal_diff_score': form_goal_diff_score,
                'home_unbeaten_streak': min(stats['home_unbeaten_streak'] / 10, 1.0),
                'away_unbeaten_streak': min(stats['away_unbeaten_streak'] / 10, 1.0),
                'overall_strength': stats['overall_win_rate'],
            }
            
            # Составной вес
            weights['composite_weight'] = np.mean([
                weights['home_win_rate'],
                weights['away_win_rate'],
                weights['form_rating'],
                weights['goal_diff_score'],
                weights['overall_strength']
            ])
            
            self.team_weights[team_id] = weights
        
        # Нормализуем составные веса к диапазону 0.3-1.0
        self._normalize_weights()
        
        logger.info(f"Рассчитаны веса для {len(self.team_weights)} команд")
        return self.team_weights
    
    def _normalize_weights(self):
        """Нормализация весов"""
        composite_weights = [w['composite_weight'] for w in self.team_weights.values()]
        
        if composite_weights:
            min_weight, max_weight = min(composite_weights), max(composite_weights)
            
            for team_id in self.team_weights:
                if max_weight > min_weight:
                    normalized = 0.3 + 0.7 * (self.team_weights[team_id]['composite_weight'] - min_weight) / (max_weight - min_weight)
                    self.team_weights[team_id]['normalized_weight'] = normalized
                else:
                    self.team_weights[team_id]['normalized_weight'] = 0.65
    
    def get_team_features_for_match(self, home_team_id, away_team_id):
        """
        Возвращает признаки для конкретного матча
        """
        home_stats = self.team_stats.get(home_team_id)
        away_stats = self.team_stats.get(away_team_id)
        
        if not home_stats or not away_stats:
            return None
        
        home_weights = self.team_weights.get(home_team_id, {})
        away_weights = self.team_weights.get(away_team_id, {})
        
        features = {
            # Домашняя команда
            'home_win_rate': home_stats['home_win_rate'],
            'home_clean_sheets_rate': home_stats['home_clean_sheets_rate'],
            'home_unbeaten_streak': home_stats['home_unbeaten_streak'],
            'home_form_rating': home_stats['form_rating'],
            'home_goal_diff_per_game': home_stats['home_goal_diff_per_game'],
            'home_form_goal_diff': home_stats['form_goal_diff'],
            'home_avg_scored': home_stats['home_avg_scored'],
            'home_avg_conceded': home_stats['home_avg_conceded'],
            
            # Гостевая команда
            'away_win_rate': away_stats['away_win_rate'],
            'away_clean_sheets_rate': away_stats['away_clean_sheets_rate'],
            'away_unbeaten_streak': away_stats['away_unbeaten_streak'],
            'away_form_rating': away_stats['form_rating'],
            'away_goal_diff_per_game': away_stats['away_goal_diff_per_game'],
            'away_form_goal_diff': away_stats['form_goal_diff'],
            'away_avg_scored': away_stats['away_avg_scored'],
            'away_avg_conceded': away_stats['away_avg_conceded'],
            
            # Разницы
            'win_rate_diff': home_stats['home_win_rate'] - away_stats['away_win_rate'],
            'form_diff': home_stats['form_rating'] - away_stats['form_rating'],
            'goal_diff_diff': home_stats['goal_diff_per_game'] - away_stats['goal_diff_per_game'],
            'clean_sheets_diff': home_stats['home_clean_sheets_rate'] - away_stats['away_clean_sheets_rate'],
            
            # Веса
            'home_weight': home_weights.get('normalized_weight', 0.5),
            'away_weight': away_weights.get('normalized_weight', 0.5),
            'weight_difference': home_weights.get('normalized_weight', 0.5) - away_weights.get('normalized_weight', 0.5),
        }
        
        return features
    
    def save_results(self, filename='team_strength_analysis.json'):
        """
        Сохраняет результаты анализа в JSON файл
        """
        results = {
            'analysis_date': datetime.now().isoformat(),
            'min_matches': self.min_matches,
            'recent_matches': self.recent_matches,
            'team_stats': self.team_stats,
            'team_weights': self.team_weights,
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Результаты сохранены в {filename}")
        return filename


if __name__ == '__main__':
    # Инициализация анализатора
    analyzer = TeamStrengthAnalyzer(
        min_matches=3,
        recent_matches=5  # Используем 5 матчей для формы
    )
    
    # 1. Расчет статистики команд
    print("Расчет статистики команд...")
    stats = analyzer.calculate_all_team_stats()
    
    if not stats:
        print("Не удалось рассчитать статистику")
        exit()
    
    # 2. Расчет весов команд
    print("Расчет весов команд...")
    weights = analyzer.calculate_team_weights()
    
    # 3. Сохранение результатов
    analyzer.save_results('team_strength_results.json')
    
    print(f"\nАнализ завершен!")
    print(f"Обработано команд: {len(analyzer.team_stats)}")
    
    # Вывод топ-5 команд по весу
    print("\nТоп-5 команд по силе:")
    sorted_teams = sorted(analyzer.team_weights.items(), 
                         key=lambda x: x[1]['normalized_weight'], 
                         reverse=True)[:5]
    
    for team_id, weights in sorted_teams:
        team_name = analyzer.team_stats[team_id]['team_name']
        print(f"  {team_name}: {weights['normalized_weight']:.3f}")