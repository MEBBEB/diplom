"""
predictor.py
Футбольный предиктор с расширенными признаками формы
"""

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import json
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML библиотеки
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import xgboost as xgb
import lightgbm as lgb

np.random.seed(42)

DB_SETTINGS = {
    'database': 'football_stats',
    'user': 'postgres',
    'password': 'Goyda_man1',
    'host': 'localhost',
    'port': '5432'
}


class FootballPredictor:
    """
    Футбольный предиктор с расширенными признаками формы
    """
    
    def __init__(self):
        self.models = {}
        self.total_model = None
        self.total_scaler = StandardScaler()
        self.scaler = StandardScaler()
        self.feature_names = []
        self.classical_stats = {}
        self.bayesian_stats = {}
        self.team_names = {}
        self.h2h_stats = {}
        self.league_stats = {}
        self.best_accuracy = 0
        self.best_total_accuracy = 0
        self.is_trained = False
        
        os.makedirs('models', exist_ok=True)
        
    def load_stats(self):
        """Загрузка готовых статистик из JSON файлов"""
        if os.path.exists('team_strength_results.json'):
            with open('team_strength_results.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.classical_stats = {int(k): v for k, v in data.get('team_stats', {}).items()}
        
        if os.path.exists('bayesian_results.json'):
            with open('bayesian_results.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.bayesian_stats = {int(k): v for k, v in data.get('enhanced_stats', {}).items()}
        
        for team_id, stats in self.classical_stats.items():
            self.team_names[int(team_id)] = stats.get('team_name', f"Team_{team_id}")
    
    def _load_h2h_statistics(self):
        """Загрузка статистики личных встреч"""
        conn = None
        try:
            conn = psycopg2.connect(**DB_SETTINGS)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            cursor.execute("""
                SELECT 
                    LEAST(home_team_id, away_team_id) as team1_id,
                    GREATEST(home_team_id, away_team_id) as team2_id,
                    COUNT(*) as total_matches,
                    SUM(CASE WHEN ftr = 'H' AND home_team_id = LEAST(home_team_id, away_team_id) THEN 1
                            WHEN ftr = 'A' AND away_team_id = LEAST(home_team_id, away_team_id) THEN 1 ELSE 0 END) as team1_wins,
                    SUM(CASE WHEN ftr = 'D' THEN 1 ELSE 0 END) as draws,
                    SUM(CASE WHEN fthg + ftag > 2.5 THEN 1 ELSE 0 END) as over_25
                FROM matches
                GROUP BY LEAST(home_team_id, away_team_id), GREATEST(home_team_id, away_team_id)
                HAVING COUNT(*) >= 2
            """)
            
            for row in cursor.fetchall():
                t1, t2, total, t1_wins, draws, over = row
                if total > 0:
                    # Сохраняем для пары (t1, t2)
                    self.h2h_stats[(t1, t2)] = {
                        'matches': total,  # <- ВАЖНО: сохраняем количество матчей!
                        'win_rate': t1_wins / total,
                        'draw_rate': draws / total,
                        'loss_rate': (total - t1_wins - draws) / total,
                        'over_25_rate': over / total
                    }
                    # Сохраняем для обратной пары (t2, t1)
                    self.h2h_stats[(t2, t1)] = {
                        'matches': total,  # <- ВАЖНО: сохраняем количество матчей!
                        'win_rate': (total - t1_wins - draws) / total,
                        'draw_rate': draws / total,
                        'loss_rate': t1_wins / total,
                        'over_25_rate': over / total
                    }
        except Exception as e:
            print(f"Ошибка загрузки H2H: {e}")
        finally:
            if conn:
                conn.close()
    
    def _load_league_stats(self):
        """Загрузка статистики по лигам"""
        conn = None
        try:
            conn = psycopg2.connect(**DB_SETTINGS)
            query = """
            SELECT 
                l.league_name,
                AVG(CASE WHEN m.ftr = 'H' THEN 1.0 ELSE 0 END) as home_win_rate,
                AVG(m.fthg + m.ftag) as avg_total_goals,
                AVG(m.fthg) as avg_home_goals,
                AVG(m.ftag) as avg_away_goals
            FROM matches m
            JOIN leagues l ON m.league_id = l.league_id
            WHERE m.season IN ('2022-2023', '2023-2024', '2024-2025')
            GROUP BY l.league_name
            """
            df = pd.read_sql(query, conn)
            for _, row in df.iterrows():
                self.league_stats[row['league_name']] = row.to_dict()
        except Exception as e:
            print(f"Ошибка загрузки статистики лиг: {e}")
        finally:
            if conn:
                conn.close()
    
    def _get_team_league(self, team_id):
        """Определение лиги команды"""
        conn = None
        try:
            conn = psycopg2.connect(**DB_SETTINGS)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT l.league_name
                FROM matches m
                JOIN leagues l ON m.league_id = l.league_id
                WHERE m.home_team_id = %s OR m.away_team_id = %s
                LIMIT 1
            """, (team_id, team_id))
            result = cursor.fetchone()
            return result[0] if result else None
        except:
            return None
        finally:
            if conn:
                conn.close()
    
    def _calculate_detailed_form(self, team_id):
        """Детальный расчет формы с разными периодами"""
        conn = None
        result = {
            'form_3': {'avg_points': 0, 'goal_diff': 0, 'wins': 0},
            'form_5': {'avg_points': 0, 'goal_diff': 0, 'wins': 0},
            'form_10': {'avg_points': 0, 'goal_diff': 0, 'wins': 0},
            'trend': 0,
            'scoring_streak': 0,
            'clean_sheets_streak': 0,
        }
        
        try:
            conn = psycopg2.connect(**DB_SETTINGS)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
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
                    END as scored,
                    CASE 
                        WHEN home_team_id = %s THEN ftag
                        ELSE fthg
                    END as conceded
                FROM matches
                WHERE (home_team_id = %s OR away_team_id = %s)
                ORDER BY match_date DESC
                LIMIT 15
            """, [team_id, team_id, team_id, team_id, team_id])
            
            matches = cursor.fetchall()
            
            if len(matches) >= 3:
                last_3 = matches[:3]
                result['form_3']['avg_points'] = sum(m[0] for m in last_3) / 3
                result['form_3']['goal_diff'] = sum(m[1] - m[2] for m in last_3)
                result['form_3']['wins'] = sum(1 for m in last_3 if m[0] == 3)
            
            if len(matches) >= 5:
                last_5 = matches[:5]
                result['form_5']['avg_points'] = sum(m[0] for m in last_5) / 5
                result['form_5']['goal_diff'] = sum(m[1] - m[2] for m in last_5)
                result['form_5']['wins'] = sum(1 for m in last_5 if m[0] == 3)
            
            if len(matches) >= 10:
                last_10 = matches[:10]
                result['form_10']['avg_points'] = sum(m[0] for m in last_10) / 10
                result['form_10']['goal_diff'] = sum(m[1] - m[2] for m in last_10)
                result['form_10']['wins'] = sum(1 for m in last_10 if m[0] == 3)
            
            if len(matches) >= 6:
                last_3_avg = sum(m[0] for m in matches[:3]) / 3
                prev_3_avg = sum(m[0] for m in matches[3:6]) / 3
                result['trend'] = last_3_avg - prev_3_avg
            
            if matches:
                for m in matches:
                    if m[1] > 0:
                        result['scoring_streak'] += 1
                    else:
                        break
                
                for m in matches:
                    if m[2] == 0:
                        result['clean_sheets_streak'] += 1
                    else:
                        break
            
        except Exception as e:
            print(f"Ошибка расчета формы: {e}")
        finally:
            if conn:
                conn.close()
        
        return result
    
    def prepare_training_data(self):
        """Подготовка данных для обучения"""
        conn = None
        try:
            conn = psycopg2.connect(**DB_SETTINGS)
            
            df = pd.read_sql("""
                SELECT 
                    m.home_team_id,
                    m.away_team_id,
                    m.fthg,
                    m.ftag,
                    m.ftr,
                    l.league_name
                FROM matches m
                JOIN leagues l ON m.league_id = l.league_id
                WHERE m.season IN ('2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025')
                ORDER BY m.match_date
            """, conn)
            
            df['fthg'] = pd.to_numeric(df['fthg'], errors='coerce').fillna(0)
            df['ftag'] = pd.to_numeric(df['ftag'], errors='coerce').fillna(0)
            df['outcome'] = 0
            df.loc[df['ftr'] == 'H', 'outcome'] = 1
            df.loc[df['ftr'] == 'A', 'outcome'] = 2
            df['over_2_5'] = ((df['fthg'] + df['ftag']) > 2.5).astype(int)
            
            self.feature_names = [
                # 1-4: home stats
                'home_win_rate', 'home_form', 'home_goal_diff', 'home_avg_scored',
                # 5-8: away stats
                'away_win_rate', 'away_form', 'away_goal_diff', 'away_avg_conceded',
                # 9-13: differences
                'win_rate_diff', 'form_diff', 'goal_diff_diff', 'avg_scored_diff', 'defense_diff',
                # 14-16: bayesian
                'bayesian_home_win', 'bayesian_away_win', 'bayesian_home_advantage',
                # 17-19: h2h
                'h2h_win_rate', 'h2h_draw_rate', 'h2h_over_rate',
                # 20-22: form_3
                'form_3_avg_points', 'form_3_goal_diff', 'form_3_wins',
                # 23-25: form_5
                'form_5_avg_points', 'form_5_goal_diff', 'form_5_wins',
                # 26-28: form_10
                'form_10_avg_points', 'form_10_goal_diff', 'form_10_wins',
                # 29-31: form trends
                'form_trend', 'scoring_streak', 'clean_sheets_streak',
                # 32-35: league stats (4 признака!)
                'league_avg_goals', 'league_home_win_rate', 'league_avg_home_goals', 'league_avg_away_goals'
            ]
            
            X, y_outcome, y_total = [], [], []
            skipped = 0
            
            for _, match in df.iterrows():
                h_id, a_id = match['home_team_id'], match['away_team_id']
                league = match['league_name']
                
                h_class = self.classical_stats.get(h_id, {})
                a_class = self.classical_stats.get(a_id, {})
                h_bayes = self.bayesian_stats.get(h_id, {})
                a_bayes = self.bayesian_stats.get(a_id, {})
                
                if not h_class or not a_class:
                    skipped += 1
                    continue
                
                h2h = self.h2h_stats.get((h_id, a_id), {})
                league_stat = self.league_stats.get(league, {})
                h_form = self._calculate_detailed_form(h_id)
                
                h_base_form = h_class.get('form_rating', 0.5)
                a_base_form = a_class.get('form_rating', 0.5)
                
                # 35 признаков в правильном порядке
                features = [
                    # 1-4: home stats
                    float(h_class.get('home_win_rate', 0.5)),
                    h_base_form,
                    float(h_class.get('goal_diff_per_game', 0)),
                    float(h_class.get('home_avg_scored', 1.0)) / 3,
                    
                    # 5-8: away stats
                    float(a_class.get('away_win_rate', 0.5)),
                    a_base_form,
                    float(a_class.get('goal_diff_per_game', 0)),
                    float(a_class.get('away_avg_conceded', 1.0)) / 3,
                    
                    # 9-13: differences (5)
                    float(h_class.get('home_win_rate', 0.5)) - float(a_class.get('away_win_rate', 0.5)),
                    h_base_form - a_base_form,
                    float(h_class.get('goal_diff_per_game', 0)) - float(a_class.get('goal_diff_per_game', 0)),
                    float(h_class.get('home_avg_scored', 1.0)) - float(a_class.get('away_avg_scored', 1.0)),
                    float(h_class.get('home_avg_conceded', 1.0)) - float(a_class.get('away_avg_conceded', 1.0)),
                    
                    # 14-16: bayesian (3)
                    float(h_bayes.get('bayesian_win_prob', 0.5)),
                    float(a_bayes.get('bayesian_win_prob', 0.5)),
                    float(h_bayes.get('bayesian_home_advantage', 0.2)),
                    
                    # 17-19: h2h (3)
                    float(h2h.get('win_rate', 0.5)),
                    float(h2h.get('draw_rate', 0.3)),
                    float(h2h.get('over_25_rate', 0.5)),
                    
                    # 20-22: form_3 (3)
                    h_form['form_3']['avg_points'] / 3,
                    min(max(h_form['form_3']['goal_diff'] / 10, -1), 1),
                    h_form['form_3']['wins'] / 3,
                    
                    # 23-25: form_5 (3)
                    h_form['form_5']['avg_points'] / 3,
                    min(max(h_form['form_5']['goal_diff'] / 10, -1), 1),
                    h_form['form_5']['wins'] / 5,
                    
                    # 26-28: form_10 (3)
                    h_form['form_10']['avg_points'] / 3,
                    min(max(h_form['form_10']['goal_diff'] / 20, -1), 1),
                    h_form['form_10']['wins'] / 10,
                    
                    # 29-31: form trends (3)
                    min(max(h_form['trend'] / 3, -1), 1),
                    min(h_form['scoring_streak'] / 10, 1),
                    min(h_form['clean_sheets_streak'] / 5, 1),
                    
                    # 32-35: league stats (4 признака!)
                    float(league_stat.get('avg_total_goals', 2.5)) / 5,
                    float(league_stat.get('home_win_rate', 0.45)),
                    float(league_stat.get('avg_home_goals', 1.3)) / 3,
                    float(league_stat.get('avg_away_goals', 1.2)) / 3
                ]
                
                X.append(features)
                y_outcome.append(match['outcome'])
                y_total.append(match['over_2_5'])
            
            print(f"Подготовлено {len(X)} матчей, признаков: {len(features)}")
            return np.array(X, dtype=np.float32), np.array(y_outcome, dtype=np.int32), np.array(y_total, dtype=np.int32)
            
        except Exception as e:
            print(f"Ошибка подготовки данных: {e}")
            return None, None, None
        finally:
            if conn:
                conn.close()
    
    def train(self):
        """Обучение моделей"""
        X, y_outcome, y_total = self.prepare_training_data()
        
        if X is None or len(X) == 0:
            print("Ошибка: нет данных для обучения")
            return False
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_outcome_train, y_outcome_test = y_outcome[:split_idx], y_outcome[split_idx:]
        y_total_train, y_total_test = y_total[:split_idx], y_total[split_idx:]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        classes = np.unique(y_outcome_train)
        weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_outcome_train)
        sample_weights = np.array([weights[y] for y in y_outcome_train])
        
        xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
        xgb_model.fit(X_train_scaled, y_outcome_train, sample_weight=sample_weights)
        
        lgb_model = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, 
                                      class_weight='balanced', random_state=42, verbose=-1)
        lgb_model.fit(X_train_scaled, y_outcome_train, sample_weight=sample_weights)
        
        rf_model = RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced',
                                         random_state=42, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_outcome_train)
        
        lr_model = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
        lr_model.fit(X_train_scaled, y_outcome_train)
        
        self.models['ensemble'] = VotingClassifier(
            estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('rf', rf_model), ('lr', lr_model)],
            voting='soft', weights=[2.0, 2.0, 1.5, 1.0]
        )
        self.models['ensemble'].fit(X_train_scaled, y_outcome_train)
        
        y_pred = self.models['ensemble'].predict(X_test_scaled)
        self.best_accuracy = accuracy_score(y_outcome_test, y_pred)
        
        X_train_total_scaled = self.total_scaler.fit_transform(X_train)
        X_test_total_scaled = self.total_scaler.transform(X_test)
        
        self.total_model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42)
        self.total_model.fit(X_train_total_scaled, y_total_train)
        
        y_total_pred = self.total_model.predict(X_test_total_scaled)
        self.best_total_accuracy = accuracy_score(y_total_test, y_total_pred)
        
        self.is_trained = True
        
        with open('models/predictor_latest.pkl', 'wb') as f:
            pickle.dump({
                'ensemble': self.models['ensemble'],
                'total_model': self.total_model,
                'scaler': self.scaler,
                'total_scaler': self.total_scaler,
                'team_names': self.team_names,
                'feature_names': self.feature_names,
                'accuracy': self.best_accuracy,
                'total_accuracy': self.best_total_accuracy
            }, f)
        
        return True
    
    def load_model(self):
        """Загрузка модели"""
        path = 'models/predictor_latest.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.models['ensemble'] = data['ensemble']
            self.total_model = data['total_model']
            self.scaler = data['scaler']
            self.total_scaler = data['total_scaler']
            self.team_names = data['team_names']
            self.feature_names = data['feature_names']
            self.best_accuracy = data['accuracy']
            self.best_total_accuracy = data['total_accuracy']
            self.is_trained = True
            return True
        return False
    
    def predict(self, home_id, away_id):
        """Предсказание матча"""
        h_class = self.classical_stats.get(home_id, {})
        a_class = self.classical_stats.get(away_id, {})
        
        if not h_class or not a_class:
            return None
        
        h_bayes = self.bayesian_stats.get(home_id, {})
        a_bayes = self.bayesian_stats.get(away_id, {})
        h2h = self.h2h_stats.get((home_id, away_id), {})
        league = self._get_team_league(home_id)
        league_stat = self.league_stats.get(league, {})
        h_form = self._calculate_detailed_form(home_id)
        
        h_base = h_class.get('form_rating', 0.5)
        a_base = a_class.get('form_rating', 0.5)
        
        # 35 признаков в ТОЧНО таком же порядке, как в prepare_training_data
        features = [[
            # 1-4: home stats
            float(h_class.get('home_win_rate', 0.5)),
            h_base,
            float(h_class.get('goal_diff_per_game', 0)),
            float(h_class.get('home_avg_scored', 1.0)) / 3,
            
            # 5-8: away stats
            float(a_class.get('away_win_rate', 0.5)),
            a_base,
            float(a_class.get('goal_diff_per_game', 0)),
            float(a_class.get('away_avg_conceded', 1.0)) / 3,
            
            # 9-13: differences
            float(h_class.get('home_win_rate', 0.5)) - float(a_class.get('away_win_rate', 0.5)),
            h_base - a_base,
            float(h_class.get('goal_diff_per_game', 0)) - float(a_class.get('goal_diff_per_game', 0)),
            float(h_class.get('home_avg_scored', 1.0)) - float(a_class.get('away_avg_scored', 1.0)),
            float(h_class.get('home_avg_conceded', 1.0)) - float(a_class.get('away_avg_conceded', 1.0)),
            
            # 14-16: bayesian
            float(h_bayes.get('bayesian_win_prob', 0.5)),
            float(a_bayes.get('bayesian_win_prob', 0.5)),
            float(h_bayes.get('bayesian_home_advantage', 0.2)),
            
            # 17-19: h2h
            float(h2h.get('win_rate', 0.5)),
            float(h2h.get('draw_rate', 0.3)),
            float(h2h.get('over_25_rate', 0.5)),
            
            # 20-22: form_3
            h_form['form_3']['avg_points'] / 3,
            min(max(h_form['form_3']['goal_diff'] / 10, -1), 1),
            h_form['form_3']['wins'] / 3,
            
            # 23-25: form_5
            h_form['form_5']['avg_points'] / 3,
            min(max(h_form['form_5']['goal_diff'] / 10, -1), 1),
            h_form['form_5']['wins'] / 5,
            
            # 26-28: form_10
            h_form['form_10']['avg_points'] / 3,
            min(max(h_form['form_10']['goal_diff'] / 20, -1), 1),
            h_form['form_10']['wins'] / 10,
            
            # 29-31: form trends
            min(max(h_form['trend'] / 3, -1), 1),
            min(h_form['scoring_streak'] / 10, 1),
            min(h_form['clean_sheets_streak'] / 5, 1),
            
            # 32-35: league stats (4 признака!)
            float(league_stat.get('avg_total_goals', 2.5)) / 5,
            float(league_stat.get('home_win_rate', 0.45)),
            float(league_stat.get('avg_home_goals', 1.3)) / 3,
            float(league_stat.get('avg_away_goals', 1.2)) / 3
        ]]
        
        X_scaled = self.scaler.transform(features)
        proba = self.models['ensemble'].predict_proba(X_scaled)[0]
        
        X_total_scaled = self.total_scaler.transform(features)
        total_proba = self.total_model.predict_proba(X_total_scaled)[0]
        
        return {
            'home': self.team_names.get(home_id, str(home_id)),
            'away': self.team_names.get(away_id, str(away_id)),
            'probs': {
                '1': proba[1],
                'X': proba[0],
                '2': proba[2]
            },
            'outcome': ['Draw', 'Home Win', 'Away Win'][np.argmax(proba)],
            'total': {
                'pred': 'Over 2.5' if total_proba[1] > 0.5 else 'Under 2.5',
                'over': total_proba[1],
                'under': total_proba[0]
            }
        }
    
    def analyze_match(self, home_id, away_id):
        """Детальный анализ матча с объяснением причин"""
        result = self.predict(home_id, away_id)
        if not result:
            return None
        
        # Получаем детальные данные для анализа
        h_class = self.classical_stats.get(home_id, {})
        a_class = self.classical_stats.get(away_id, {})
        h2h = self.h2h_stats.get((home_id, away_id), {})
        h_form = self._calculate_detailed_form(home_id)
        a_form = self._calculate_detailed_form(away_id)
        
        # Статистика текущего сезона (2025-2026)
        h_current = self._get_current_season_stats(home_id, '2025-2026')
        a_current = self._get_current_season_stats(away_id, '2025-2026')
        
        home_prob = result['probs']['1'] * 100
        draw_prob = result['probs']['X'] * 100
        away_prob = result['probs']['2'] * 100
        
        print(f"\n{'='*60}")
        print(f"АНАЛИЗ МАТЧА: {result['home']} vs {result['away']}")
        print(f"{'='*60}")
        print(f"ПРОГНОЗ:")
        print(f"   П1: {home_prob:.1f}% | X: {draw_prob:.1f}% | П2: {away_prob:.1f}%")
        print(f"   Исход: {result['outcome']}")
        print(f"   Тотал: {result['total']['pred']} ({result['total']['over']*100:.1f}% / {result['total']['under']*100:.1f}%)")
        
        # 1. Анализ формы
        print(f"\nФОРМА КОМАНД:")
        print(f"   {result['home']}:")
        print(f"     За 5 матчей: {h_form['form_5']['avg_points']:.1f} очков/матч")
        print(f"     Разница голов: {h_form['form_5']['goal_diff']:+d}")
        print(f"     Тренд: {'рост' if h_form['trend'] > 0 else 'спад'} ({h_form['trend']:+.2f})")
        print(f"     Забивает в {h_form['scoring_streak']} матчах подряд")
        
        print(f"\n   {result['away']}:")
        print(f"     За 5 матчей: {a_form['form_5']['avg_points']:.1f} очков/матч")
        print(f"     Разница голов: {a_form['form_5']['goal_diff']:+d}")
        print(f"     Тренд: {'рост' if a_form['trend'] > 0 else 'спад'} ({a_form['trend']:+.2f})")
        print(f"     Забивает в {a_form['scoring_streak']} матчах подряд")
        
        # 2. Сравнение ключевых показателей
        print(f"\nСРАВНЕНИЕ СИЛ:")
        
        # Историческая статистика (вся история)
        h_home_hist = h_class.get('home_win_rate', 0.5) * 100
        a_away_hist = a_class.get('away_win_rate', 0.5) * 100
        
        # Статистика текущего сезона
        h_home_curr = h_current['home_win_rate'] * 100
        a_away_curr = a_current['away_win_rate'] * 100
        
        print(f"ДОМА:")
        print(f"     {result['home']} (вся история): {h_home_hist:.0f}% побед дома")
        if h_current['home_matches'] > 0:
            print(f"     {result['home']} (текущий сезон): {h_home_curr:.0f}% побед дома ({h_current['home_matches']} матчей)")
        
        print(f"\nВ ГОСТЯХ:")
        print(f"    {result['away']} (вся история): {a_away_hist:.0f}% побед в гостях")
        if a_current['away_matches'] > 0:
            print(f"     {result['away']} (текущий сезон): {a_away_curr:.0f}% побед в гостях ({a_current['away_matches']} матчей)")
        
        h_gd = h_class.get('goal_diff_per_game', 0)
        a_gd = a_class.get('goal_diff_per_game', 0)
        print(f"\n   РАЗНИЦА ГОЛОВ:")
        print(f"     {result['home']}: {h_gd:+.2f} за матч")
        print(f"     {result['away']}: {a_gd:+.2f} за матч")
        
        # 3. Личные встречи
        if h2h:
            print(f"\nЛИЧНЫЕ ВСТРЕЧИ ({h2h.get('matches', 0)} матчей):")
            print(f"   {result['home']} побед: {h2h.get('win_rate', 0.5)*100:.1f}%")
            print(f"   Ничьи: {h2h.get('draw_rate', 0.3)*100:.1f}%")
            print(f"   {result['away']} побед: {h2h.get('loss_rate', 0.2)*100:.1f}%")
        
        # 4. КЛЮЧЕВЫЕ ФАКТОРЫ - ОБЪЯСНЕНИЕ ПОЧЕМУ
        reasons = []
        counter_reasons = []
        
        # Форма за 5 матчей
        if h_form['form_5']['avg_points'] > a_form['form_5']['avg_points'] + 0.3:
            reasons.append(f"лучшая форма в последних матчах ({h_form['form_5']['avg_points']:.1f} vs {a_form['form_5']['avg_points']:.1f})")
        elif a_form['form_5']['avg_points'] > h_form['form_5']['avg_points'] + 0.3:
            counter_reasons.append(f"соперник в лучшей форме")
        
        # Тренд формы
        if h_form['trend'] > 0.5 and a_form['trend'] < -0.3:
            reasons.append("форма растет, у соперника падает")
        elif a_form['trend'] > 0.5 and h_form['trend'] < -0.3:
            counter_reasons.append("форма соперника растет")
        
        # Домашнее преимущество в текущем сезоне
        if h_current['home_win_rate'] > 0.6 and h_current['home_matches'] > 3:
            reasons.append(f"отличная домашняя форма в текущем сезоне ({h_current['home_win_rate']*100:.0f}%)")
        
        # Проблемы на выезде в текущем сезоне
        if a_current['away_win_rate'] < 0.2 and a_current['away_matches'] > 3:
            counter_reasons.append(f"слабая выездная форма в текущем сезоне")
        
        # Серии
        if h_form['scoring_streak'] > 5:
            reasons.append(f"забивает в {h_form['scoring_streak']} матчах подряд")
        if h_form['clean_sheets_streak'] > 2:
            reasons.append(f"{h_form['clean_sheets_streak']} сухих матчей подряд")
        
        # Личные встречи
        if h2h and h2h.get('win_rate', 0.5) > 0.6 and h2h.get('matches', 0) > 3:
            reasons.append("историческое преимущество в личных встречах")
        
        # Разница голов
        if h_gd > a_gd + 0.5:
            reasons.append(f"лучшая разница голов ({h_gd:+.2f} vs {a_gd:+.2f})")
        elif a_gd > h_gd + 0.5:
            counter_reasons.append(f"соперник имеет лучшую разницу голов")
        
        print(f"\n{'='*60}")
        print("💡 ВЫВОД:")
        
        if home_prob > 60:
            print(f"   Модель уверена в победе {result['home']} потому что:")
            for r in reasons[:3]:
                print(f"      • {r}")
        elif away_prob > 60:
            print(f"   Модель ожидает победу {result['away']} потому что:")
            for r in counter_reasons[:3]:
                print(f"      • {r}")
        elif max(home_prob, away_prob) > 45:
            print(f"   Модель ожидает равный матч. Ключевые факторы:")
            if reasons:
                print(f"      За {result['home']}: {reasons[0]}")
            if counter_reasons:
                print(f"      За {result['away']}: {counter_reasons[0]}")
        else:
            print(f"   Неопределенный прогноз - слишком много противоречивых факторов")
        
        print(f"{'='*60}")
        return result
    
    def get_feature_importance(self):
        """Важность признаков"""
        if not self.is_trained:
            return
        
        importance = {}
        for name, model in self.models['ensemble'].named_estimators_.items():
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                for i, feat in enumerate(self.feature_names):
                    importance[feat] = importance.get(feat, 0) + imp[i]
        
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("\nТОП-10 ВАЖНЫХ ПРИЗНАКОВ:")
        desc = {
            'h2h_win_rate': 'Личные встречи (победы)',
            'h2h_draw_rate': 'Личные встречи (ничьи)',
            'goal_diff_diff': 'Разница в разнице голов',
            'win_rate_diff': 'Разница в победах',
            'form_diff': 'Разница в форме',
            'bayesian_home_advantage': 'Преимущество своего поля',
            'form_5_avg_points': 'Форма за 5 матчей',
            'form_trend': 'Тренд формы',
            'scoring_streak': 'Серия с голами',
            'league_avg_goals': 'Средняя результативность лиги'
        }
        
        for i, (feat, val) in enumerate(sorted_imp, 1):
            name = desc.get(feat, feat.replace('_', ' '))
            print(f"   {i}. {name}: {val*100:.1f}%")

    def _get_current_season_stats(self, team_id, season='2025-2026'):
        """Получение статистики команды в текущем сезоне"""
        conn = None
        try:
            conn = psycopg2.connect(**DB_SETTINGS)
            cursor = conn.cursor()
            
            # Домашние матчи в текущем сезоне
            cursor.execute("""
                SELECT 
                    COUNT(*) as home_matches,
                    COUNT(CASE WHEN ftr = 'H' THEN 1 END) as home_wins
                FROM matches
                WHERE home_team_id = %s AND season = %s
            """, (team_id, season))
            home_res = cursor.fetchone()
            
            # Выездные матчи в текущем сезоне
            cursor.execute("""
                SELECT 
                    COUNT(*) as away_matches,
                    COUNT(CASE WHEN ftr = 'A' THEN 1 END) as away_wins
                FROM matches
                WHERE away_team_id = %s AND season = %s
            """, (team_id, season))
            away_res = cursor.fetchone()
            
            home_matches = home_res[0] or 0
            home_wins = home_res[1] or 0
            away_matches = away_res[0] or 0
            away_wins = away_res[1] or 0
            
            return {
                'home_win_rate': home_wins / home_matches if home_matches > 0 else 0,
                'away_win_rate': away_wins / away_matches if away_matches > 0 else 0,
                'home_matches': home_matches,
                'away_matches': away_matches
            }
        except Exception as e:
            print(f"Ошибка получения статистики сезона: {e}")
            return {'home_win_rate': 0, 'away_win_rate': 0, 'home_matches': 0, 'away_matches': 0}
        finally:
            if conn:
                conn.close()

def main():
    """Интерактивный режим"""
    print("\nФУТБОЛЬНЫЙ ПРОГНОЗЕР v2.0")
    print("="*40)
    
    predictor = FootballPredictor()
    predictor.load_stats()
    predictor._load_h2h_statistics()
    predictor._load_league_stats()
    
    if not predictor.load_model():
        print(" Модель не найдена. Запустите: python predictor.py --train")
        return
    
    print(f" Модель загружена (исход: {predictor.best_accuracy*100:.1f}%, тотал: {predictor.best_total_accuracy*100:.1f}%)")
    print("\nДоступные команды:")
    print("   <id1> <id2>      - прогноз матча")
    print("   --analyze <id1> <id2> - детальный анализ")
    print("   --importance     - важность признаков")
    print("   q                - выход")
    
    while True:
        try:
            cmd = input("\n> ").strip()
            if cmd == 'q':
                break
            
            if cmd == '--importance':
                predictor.get_feature_importance()
                continue
            
            if cmd.startswith('--analyze'):
                parts = cmd.split()
                if len(parts) == 3:
                    res = predictor.analyze_match(int(parts[1]), int(parts[2]))
                    if not res:
                        print(" Команды не найдены")
                else:
                    print(" Формат: --analyze <id1> <id2>")
                continue
            
            parts = cmd.split()
            if len(parts) == 2:
                res = predictor.predict(int(parts[0]), int(parts[1]))
                if res:
                    print(f"\n{res['home']} vs {res['away']}")
                    print(f"П1: {res['probs']['1']*100:.1f}% | X: {res['probs']['X']*100:.1f}% | П2: {res['probs']['2']*100:.1f}%")
                    print(f"→ {res['outcome']} | {res['total']['pred']}")
                else:
                    print("Команды не найдены")
            else:
                print("Неизвестная команда")
                    
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Ошибка: {e}")


def train():
    """Режим обучения"""
    print("\nОБУЧЕНИЕ МОДЕЛИ")
    print("="*40)
    
    predictor = FootballPredictor()
    predictor.load_stats()
    predictor._load_h2h_statistics()
    predictor._load_league_stats()
    
    if predictor.train():
        print("\nМодель успешно обучена!")
        predictor.get_feature_importance()
    else:
        print("Ошибка обучения")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--train':
        train()
    else:
        main()