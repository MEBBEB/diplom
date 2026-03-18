"""
bayesian_analyzer.py
Байесовское обновление силы команд с использованием структуры БД
"""

import numpy as np
import pandas as pd
import psycopg2
import json
from datetime import datetime, timedelta, date
from scipy.stats import beta, norm, poisson
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

class BayesianTeamStrengthUpdater:
    """
    Байесовское обновление силы команд на основе результатов матчей
    
    Теоретическая основа:
    - Априорное распределение: Beta(α, β) для вероятности победы
    - Правдоподобие: Биномиальное распределение
    - Апостериорное распределение: Beta(α + победы, β + поражения)
    """
    
    def __init__(self, alpha_prior=2, beta_prior=2, 
                 attack_prior_mean=1.0, attack_prior_std=0.5,
                 defense_prior_mean=1.0, defense_prior_std=0.5,
                 decay_factor=0.95, min_matches=5):
        """
        Инициализация байесовского обновления
        """
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.attack_prior_mean = attack_prior_mean
        self.attack_prior_std = attack_prior_std
        self.defense_prior_mean = defense_prior_mean
        self.defense_prior_std = defense_prior_std
        self.decay_factor = decay_factor
        self.min_matches = min_matches
        
        # Хранилища данных
        self.team_posteriors = {}  # Апостериорные распределения
        self.team_history = {}      # История матчей
        self.team_names = {}        # Названия команд
        
    def initialize_team(self, team_id, team_name):
        """
        Инициализация априорных распределений для новой команды
        """
        self.team_posteriors[team_id] = {
            'team_id': team_id,
            'team_name': team_name,
            
            # Beta-распределение для вероятности победы
            'win_alpha': self.alpha_prior,
            'win_beta': self.beta_prior,
            
            # Нормальные распределения для атаки и защиты
            'attack_mean': self.attack_prior_mean,
            'attack_std': self.attack_prior_std,
            'defense_mean': self.defense_prior_mean,
            'defense_std': self.defense_prior_std,
            
            # Преимущество своего поля
            'home_advantage_mean': 0.2,
            'home_advantage_std': 0.1,
            
            # Счетчики
            'total_matches': 0,
            'home_matches': 0,
            'away_matches': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_scored': 0,
            'goals_conceded': 0,
            'last_update': None  # Будем хранить как datetime или None
        }
        
        self.team_history[team_id] = []
        self.team_names[team_id] = team_name
        
    def _calculate_time_weight(self, match_date, last_update):
        """
        Расчет веса матча на основе времени
        
        Args:
            match_date: дата матча (datetime.date или datetime.datetime)
            last_update: дата последнего обновления (datetime или None)
        """
        if last_update is None:
            return 1.0
        
        # Преобразуем всё к datetime для сравнения
        if isinstance(match_date, date) and not isinstance(match_date, datetime):
            match_date = datetime.combine(match_date, datetime.min.time())
        
        if isinstance(last_update, date) and not isinstance(last_update, datetime):
            last_update = datetime.combine(last_update, datetime.min.time())
        
        # Расчет разницы в днях
        days_ago = (datetime.now() - last_update).days
        time_weight = self.decay_factor ** (days_ago / 30)  # Месячный распад
        return time_weight
        
    def update_with_match(self, team_id, is_home, opponent_id, 
                          goals_scored, goals_conceded, match_date, 
                          match_weight=1.0):
        """
        Байесовское обновление на основе одного матча
        """
        if team_id not in self.team_posteriors:
            logger.warning(f"Команда {team_id} не инициализирована")
            return
        
        posteriors = self.team_posteriors[team_id]
        
        # Вес матча с учетом времени
        time_weight = self._calculate_time_weight(match_date, posteriors['last_update'])
        total_weight = match_weight * time_weight
        
        # 1. Обновление вероятности победы (Beta-распределение)
        if goals_scored > goals_conceded:  # Победа
            posteriors['win_alpha'] += total_weight
            posteriors['wins'] += 1
        elif goals_scored < goals_conceded:  # Поражение
            posteriors['win_beta'] += total_weight
            posteriors['losses'] += 1
        else:  # Ничья (считаем как полпобеды)
            posteriors['win_alpha'] += 0.5 * total_weight
            posteriors['win_beta'] += 0.5 * total_weight
            posteriors['draws'] += 1
        
        # 2. Обновление силы атаки
        opponent_posteriors = self.team_posteriors.get(opponent_id)
        if opponent_posteriors:
            opponent_defense = opponent_posteriors.get('defense_mean', 1.0)
            
            # Наблюдаемая сила атаки с поправкой на защиту соперника
            observed_attack = goals_scored / max(opponent_defense, 0.3)
            
            # Байесовское обновление нормального распределения
            prior_precision = 1 / posteriors['attack_std']**2
            likelihood_precision = total_weight
            
            posterior_precision = prior_precision + likelihood_precision
            posterior_mean = (
                (posteriors['attack_mean'] * prior_precision) + 
                (observed_attack * likelihood_precision)
            ) / posterior_precision
            
            posteriors['attack_mean'] = posterior_mean
            posteriors['attack_std'] = 1 / np.sqrt(posterior_precision)
        
        # 3. Обновление силы защиты
        if opponent_posteriors:
            opponent_attack = opponent_posteriors.get('attack_mean', 1.0)
            
            # Наблюдаемая сила защиты
            observed_defense = goals_conceded / max(opponent_attack, 0.3)
            
            prior_precision = 1 / posteriors['defense_std']**2
            likelihood_precision = total_weight
            
            posterior_precision = prior_precision + likelihood_precision
            posterior_mean = (
                (posteriors['defense_mean'] * prior_precision) + 
                (observed_defense * likelihood_precision)
            ) / posterior_precision
            
            posteriors['defense_mean'] = posterior_mean
            posteriors['defense_std'] = 1 / np.sqrt(posterior_precision)
        
        # 4. Обновление преимущества своего поля
        if is_home:
            # Наблюдаемое преимущество
            expected_no_home = (posteriors['attack_mean'] - posteriors['defense_mean'])
            observed_advantage = (goals_scored - goals_conceded) - expected_no_home
            
            prior_precision = 1 / posteriors['home_advantage_std']**2
            posterior_precision = prior_precision + total_weight
            posterior_mean = (
                (posteriors['home_advantage_mean'] * prior_precision) + 
                (observed_advantage * total_weight)
            ) / posterior_precision
            
            posteriors['home_advantage_mean'] = posterior_mean
            posteriors['home_advantage_std'] = 1 / np.sqrt(posterior_precision)
        
        # Обновление счетчиков
        posteriors['total_matches'] += 1
        if is_home:
            posteriors['home_matches'] += 1
        else:
            posteriors['away_matches'] += 1
            
        posteriors['goals_scored'] += goals_scored
        posteriors['goals_conceded'] += goals_conceded
        
        # Сохраняем дату последнего обновления
        if isinstance(match_date, date) and not isinstance(match_date, datetime):
            posteriors['last_update'] = datetime.combine(match_date, datetime.min.time())
        else:
            posteriors['last_update'] = match_date
        
        # Сохраняем в историю
        self.team_history[team_id].append({
            'date': match_date,
            'opponent': opponent_id,
            'is_home': is_home,
            'goals_scored': goals_scored,
            'goals_conceded': goals_conceded,
            'weight': total_weight
        })
    
    def predict_match_outcome(self, home_id, away_id, n_simulations=40000):
        """
        Предсказание исхода матча методом Монте-Карло
        """
        if home_id not in self.team_posteriors or away_id not in self.team_posteriors:
            logger.warning("Одна из команд не инициализирована")
            return [0.4, 0.3, 0.3]
        
        home = self.team_posteriors[home_id]
        away = self.team_posteriors[away_id]
        
        home_wins = 0
        draws = 0
        away_wins = 0
        
        for _ in range(n_simulations):
            # Сэмплируем параметры из апостериорных распределений
            home_attack = np.random.normal(home['attack_mean'], home['attack_std'])
            home_defense = np.random.normal(home['defense_mean'], home['defense_std'])
            home_advantage = np.random.normal(home['home_advantage_mean'], home['home_advantage_std'])
            
            away_attack = np.random.normal(away['attack_mean'], away['attack_std'])
            away_defense = np.random.normal(away['defense_mean'], away['defense_std'])
            
            # Ожидаемые голы
            lambda_home = max(0.1, home_attack * away_defense * (1 + max(0, home_advantage)))
            lambda_away = max(0.1, away_attack * home_defense)
            
            # Симулируем голы
            home_goals = np.random.poisson(lambda_home)
            away_goals = np.random.poisson(lambda_away)
            
            if home_goals > away_goals:
                home_wins += 1
            elif home_goals < away_goals:
                away_wins += 1
            else:
                draws += 1
        
        return [
            home_wins / n_simulations,
            draws / n_simulations,
            away_wins / n_simulations
        ]
    
    def get_win_probability(self, team_id):
        """Вероятность победы с учетом неопределенности"""
        if team_id not in self.team_posteriors:
            return 0.5, 0.3
        
        posteriors = self.team_posteriors[team_id]
        
        # Матожидание Beta-распределения
        mean = posteriors['win_alpha'] / (posteriors['win_alpha'] + posteriors['win_beta'])
        
        # Стандартное отклонение
        std = np.sqrt(
            (posteriors['win_alpha'] * posteriors['win_beta']) / 
            ((posteriors['win_alpha'] + posteriors['win_beta'])**2 * 
             (posteriors['win_alpha'] + posteriors['win_beta'] + 1))
        )
        
        return mean, std
    
    def get_team_ranking(self, metric='win_prob'):
        """Ранжирование команд"""
        rankings = []
        
        for team_id, posteriors in self.team_posteriors.items():
            if posteriors['total_matches'] < self.min_matches:
                continue
            
            if metric == 'win_prob':
                value = posteriors['win_alpha'] / (posteriors['win_alpha'] + posteriors['win_beta'])
            elif metric == 'attack':
                value = posteriors['attack_mean']
            elif metric == 'defense':
                value = 1 / posteriors['defense_mean']
            elif metric == 'net_rating':
                value = posteriors['attack_mean'] - posteriors['defense_mean']
            else:
                continue
            
            rankings.append({
                'team_id': team_id,
                'team_name': posteriors['team_name'],
                'value': value,
                'matches': posteriors['total_matches']
            })
        
        rankings.sort(key=lambda x: x['value'], reverse=True)
        return rankings


class BayesianTeamAnalyzer:
    """
    Интеграция байесовского обновления с БД
    """
    
    def __init__(self, db_settings):
        self.db_settings = db_settings
        self.bayesian_updater = BayesianTeamStrengthUpdater()
        self.conn = None
        
    def connect(self):
        """Подключение к БД"""
        try:
            self.conn = psycopg2.connect(**self.db_settings)
            return self.conn
        except Exception as e:
            logger.error(f"Ошибка подключения: {e}")
            return None
    
    def close(self):
        """Закрытие соединения"""
        if self.conn:
            self.conn.close()
    
    def load_teams_from_db(self):
        """Загрузка всех команд из БД"""
        conn = self.connect()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT team_id, team_name 
                FROM teams 
                ORDER BY team_id
            """)
            teams = cursor.fetchall()
            
            for team_id, team_name in teams:
                self.bayesian_updater.initialize_team(team_id, team_name)
            
            logger.info(f"Загружено {len(teams)} команд")
            return teams
            
        except Exception as e:
            logger.error(f"Ошибка загрузки команд: {e}")
            return []
        finally:
            cursor.close()
            self.close()
    
    def load_matches_for_bayesian_update(self, seasons=None, start_date=None, end_date=None):
        """
        Загрузка матчей и байесовское обновление
        """
        conn = self.connect()
        if not conn:
            return
        
        try:
            # Используем курсор для построчной обработки вместо pandas
            cursor = conn.cursor()
            
            query = """
            SELECT 
                m.match_id,
                m.home_team_id,
                m.away_team_id,
                m.fthg,
                m.ftag,
                m.ftr,
                m.match_date,
                m.season,
                l.league_name
            FROM matches m
            JOIN leagues l ON m.league_id = l.league_id
            WHERE 1=1
            """
            
            params = []
            
            if seasons:
                placeholders = ','.join(['%s'] * len(seasons))
                query += f" AND m.season IN ({placeholders})"
                params.extend(seasons)
            
            if start_date:
                query += " AND m.match_date >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND m.match_date <= %s"
                params.append(end_date)
            
            query += " ORDER BY m.match_date"
            
            # Выполняем запрос
            cursor.execute(query, params)
            
            # Получаем все строки
            rows = cursor.fetchall()
            total_matches = len(rows)
            logger.info(f"Загружено {total_matches} матчей")
            
            # Обновляем последовательно
            for idx, row in enumerate(rows, 1):
                match_id, home_id, away_id, fthg, ftag, ftr, match_date, season, league_name = row
                
                # Вес матча зависит от лиги
                league_weight = 1.2 if league_name in [
                    'Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1'
                ] else 1.0
                
                # Обновление для домашней команды
                self.bayesian_updater.update_with_match(
                    team_id=home_id,
                    is_home=True,
                    opponent_id=away_id,
                    goals_scored=fthg if fthg is not None else 0,
                    goals_conceded=ftag if ftag is not None else 0,
                    match_date=match_date,
                    match_weight=league_weight
                )
                
                # Обновление для гостевой команды
                self.bayesian_updater.update_with_match(
                    team_id=away_id,
                    is_home=False,
                    opponent_id=home_id,
                    goals_scored=ftag if ftag is not None else 0,
                    goals_conceded=fthg if fthg is not None else 0,
                    match_date=match_date,
                    match_weight=league_weight
                )
                
                if idx % 100 == 0:
                    logger.info(f"Обработано {idx}/{total_matches} матчей")
            
            logger.info("Байесовское обновление завершено")
            
        except Exception as e:
            logger.error(f"Ошибка при обновлении: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if cursor:
                cursor.close()
            self.close()
    
    def get_enhanced_stats(self):
        """
        Получение расширенной статистики с байесовскими оценками
        """
        enhanced_stats = {}
        
        for team_id, posteriors in self.bayesian_updater.team_posteriors.items():
            win_prob, win_std = self.bayesian_updater.get_win_probability(team_id)
            
            # Байесовские метрики
            bayesian_metrics = {
                'team_id': team_id,
                'team_name': posteriors['team_name'],
                
                # Вероятности с неопределенностью
                'bayesian_win_prob': win_prob,
                'bayesian_win_std': win_std,
                'bayesian_attack': posteriors['attack_mean'],
                'bayesian_attack_std': posteriors['attack_std'],
                'bayesian_defense': posteriors['defense_mean'],
                'bayesian_defense_std': posteriors['defense_std'],
                'bayesian_home_advantage': posteriors['home_advantage_mean'],
                
                # Кредибельность оценки
                'bayesian_confidence': min(posteriors['total_matches'] / 20, 1.0),
                'total_matches_bayes': posteriors['total_matches'],
                'wins': posteriors['wins'],
                'draws': posteriors['draws'],
                'losses': posteriors['losses'],
                'goals_scored': posteriors['goals_scored'],
                'goals_conceded': posteriors['goals_conceded'],
                
                # Комбинированная сила
                'bayesian_strength': (
                    win_prob * 0.4 + 
                    (posteriors['attack_mean'] / (posteriors['attack_mean'] + posteriors['defense_mean'])) * 0.3 +
                    (1 - posteriors['defense_mean'] / 3) * 0.3
                ),
                
                # Доверительный интервал
                'strength_lower': max(0, win_prob - 1.96 * win_std),
                'strength_upper': min(1, win_prob + 1.96 * win_std)
            }
            
            enhanced_stats[team_id] = bayesian_metrics
        
        return enhanced_stats
    
    def save_results(self, filename='bayesian_results.json'):
        """
        Сохранение результатов
        """
        results = {
            'analysis_date': datetime.now().isoformat(),
            'parameters': {
                'alpha_prior': self.bayesian_updater.alpha_prior,
                'beta_prior': self.bayesian_updater.beta_prior,
                'decay_factor': self.bayesian_updater.decay_factor,
                'min_matches': self.bayesian_updater.min_matches
            },
            'team_posteriors': {},
            'enhanced_stats': self.get_enhanced_stats(),
            'rankings': {
                'by_win_prob': self.bayesian_updater.get_team_ranking('win_prob'),
                'by_attack': self.bayesian_updater.get_team_ranking('attack'),
                'by_defense': self.bayesian_updater.get_team_ranking('defense'),
                'by_net_rating': self.bayesian_updater.get_team_ranking('net_rating')
            }
        }
        
        # Преобразуем posteriors для JSON
        for team_id, posteriors in self.bayesian_updater.team_posteriors.items():
            results['team_posteriors'][str(team_id)] = {}
            for k, v in posteriors.items():
                if k == 'last_update' and v is not None:
                    if isinstance(v, datetime):
                        results['team_posteriors'][str(team_id)][k] = v.isoformat()
                    else:
                        results['team_posteriors'][str(team_id)][k] = str(v)
                elif k != 'team_name':
                    results['team_posteriors'][str(team_id)][k] = v
            results['team_posteriors'][str(team_id)]['team_name'] = posteriors['team_name']
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Результаты сохранены в {filename}")
        return filename


if __name__ == '__main__':
    print("="*60)
    print("БАЙЕСОВСКИЙ АНАЛИЗАТОР СИЛЫ КОМАНД")
    print("="*60)
    
    # Инициализация
    analyzer = BayesianTeamAnalyzer(DB_SETTINGS)
    
    # 1. Загрузка команд
    print("\n[1] Загрузка команд...")
    analyzer.load_teams_from_db()
    
    # 2. Загрузка матчей и обновление
    print("\n[2] Байесовское обновление на исторических данных...")
    analyzer.load_matches_for_bayesian_update(
        seasons=['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025', '2025-2026']
    )
    
    # 3. Сохранение результатов
    print("\n[3] Сохранение результатов...")
    analyzer.save_results('bayesian_results.json')
    
    # 4. Вывод топ-10 команд
    print("\n[4] Топ-10 команд по байесовскому рейтингу:")
    rankings = analyzer.bayesian_updater.get_team_ranking('net_rating')
    for i, team in enumerate(rankings[:10], 1):
        print(f"{i:2d}. {team['team_name']}: {team['value']:.3f}")
    
    print("\n" + "="*60)
    print("БАЙЕСОВСКИЙ АНАЛИЗ ЗАВЕРШЕН")
    print("="*60)