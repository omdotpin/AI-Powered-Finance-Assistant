import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import numpy as np

class FinanceAssistant:
    def __init__(self):
        self.data_file = Path("data/transactions.json")
        self.budget_file = Path("data/budgets.json")
        self.data_file.parent.mkdir(exist_ok=True)
        self.transactions = self.load_transactions()
        self.budgets = self.load_budgets()

    def load_transactions(self):
        if self.data_file.exists():
            with open(self.data_file, 'r') as f:
                return json.load(f)
        return []

    def load_budgets(self):
        if self.budget_file.exists():
            with open(self.budget_file, 'r') as f:
                return json.load(f)
        return {}

    def save_budgets(self):
        with open(self.budget_file, 'w') as f:
            json.dump(self.budgets, f)

    def set_budget(self, category, amount):
        self.budgets[category] = float(amount)
        self.save_budgets()

    def get_monthly_expenses_by_category(self, category):
        current_month = datetime.now().strftime("%Y-%m")
        monthly_expenses = sum(
            abs(t['amount']) for t in self.transactions 
            if t['category'] == category 
            and t['amount'] < 0 
            and t['date'].startswith(current_month)
        )
        return monthly_expenses

    def get_budget_status(self):
        status = {}
        for category, budget in self.budgets.items():
            spent = self.get_monthly_expenses_by_category(category)
            status[category] = {
                'budget': budget,
                'spent': spent,
                'remaining': budget - spent,
                'percentage': (spent / budget * 100) if budget > 0 else 0
            }
        return status

    def get_spending_trends(self, days=30):
        df = pd.DataFrame(self.transactions)
        if len(df) == 0:
            return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['date'])
        df['amount'] = df['amount'].astype(float)
        
        last_date = df['date'].max()
        start_date = last_date - timedelta(days=days)
        
        df = df[df['date'] >= start_date]
        daily_spending = df.groupby(['date', 'category'])['amount'].sum().reset_index()
        
        return daily_spending

    def save_transactions(self):
        with open(self.data_file, 'w') as f:
            json.dump(self.transactions, f)

    def add_transaction(self, amount, category, description, date):
        transaction = {
            'amount': float(amount),
            'category': category,
            'description': description,
            'date': date
        }
        self.transactions.append(transaction)
        self.save_transactions()

    def get_total_expenses(self):
        return sum(t['amount'] for t in self.transactions if t['amount'] < 0)

    def get_total_income(self):
        return sum(t['amount'] for t in self.transactions if t['amount'] > 0)