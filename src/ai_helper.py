import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import openai
from config import OPENAI_API_KEY
import re

openai.api_key = OPENAI_API_KEY

class FinanceAI:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)
        self.label_encoder = LabelEncoder()

    def chat_response(self, query, transactions):
        if not transactions:
            return "I don't have any transaction data to analyze yet. Please add some transactions first!"

        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'])
        
        # Financial summary
        total_spent = abs(df[df['amount'] < 0]['amount'].sum())
        total_income = df[df['amount'] > 0]['amount'].sum()
        categories = df[df['amount'] < 0].groupby('category')['amount'].sum().abs()
        recent_transactions = df.sort_values('date', ascending=False).head(5)

        # Handle basic queries
        query = query.lower()
        
        # Basic responses without OpenAI
        if any(word in query for word in ['hi', 'hello', 'hey']):
            return f"Hello! Today's overview:\nTotal spent so far: ${total_spent:.2f}\nWould you like to know more about your spending?"

        try:
            # Enhanced OpenAI context
            system_prompt = """You are a sophisticated AI financial advisor. Analyze the data and provide:
            1. Clear, specific answers about transactions and spending
            2. Actionable financial insights
            3. Professional yet friendly responses
            4. Numbers formatted as currency with 2 decimal places
            5. Specific recommendations based on spending patterns"""

            context = f"""
            Financial Overview:
            - Total Income: ${total_income:.2f}
            - Total Expenses: ${total_spent:.2f}
            - Net Balance: ${total_income - total_spent:.2f}
            
            Category Breakdown:
            {categories.to_string()}
            
            Recent Transactions:
            {recent_transactions[['date', 'category', 'amount', 'description']].to_string()}
            
            User Question: {query}
            """

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.7,
                max_tokens=300,
                presence_penalty=0.6,
                frequency_penalty=0.3
            )
            return response.choices[0].message['content']
            
        except Exception as e:
            # Fallback to basic analysis if OpenAI fails
            if 'spent most' in query:
                return self._get_highest_spending(df)
            elif 'summary' in query:
                return self._get_summary(df)
            elif any(category in query for category in ['food', 'transport', 'entertainment', 'bills', 'other']):
                category = next(cat for cat in ['food', 'transport', 'entertainment', 'bills', 'other'] if cat in query)
                return self._get_category_analysis(category, df)
            
            # Date-specific queries
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', query)
            if date_match or 'today' in query or 'yesterday' in query:
                return self._get_date_spending(query, df)
            
            # Monthly analysis
            if any(word in query for word in ['month', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']):
                return self._get_monthly_spending(query, df)
            
            return "I apologize, but I encountered an error. You can ask me about:\n- Spending on specific dates\n- Monthly analysis\n- Category-wise spending\n- Overall summary"

    def prepare_data(self, transactions):
        if not transactions:
            return None, None
            
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        
        df['category_encoded'] = self.label_encoder.fit_transform(df['category'])
        
        return df

    def predict_expenses(self, df, days_ahead=30):
        if df is None or len(df) < 10:  # Need minimum data for prediction
            return None
            
        features = ['day_of_week', 'day_of_month', 'month', 'category_encoded']
        X = df[features]
        y = df['amount']
        
        self.model.fit(X, y)
        
        # Generate future dates
        future_dates = pd.date_range(start=df['date'].max(), periods=days_ahead + 1)[1:]
        future_df = pd.DataFrame({
            'date': future_dates,
            'day_of_week': future_dates.dayofweek,
            'day_of_month': future_dates.day,
            'month': future_dates.month
        })
        
        predictions = {}
        for category in df['category'].unique():
            future_df['category_encoded'] = self.label_encoder.transform([category] * len(future_df))
            category_predictions = self.model.predict(future_df[features])
            predictions[category] = category_predictions.mean()
            
        return predictions

    def get_insights(self, transactions):
        df = self.prepare_data(transactions)
        if df is None:
            return []
            
        insights = []
        
        # Spending patterns
        category_spending = df.groupby('category')['amount'].sum()
        highest_expense = category_spending.idxmin()
        
        # Unusual transactions
        mean_by_category = df.groupby('category')['amount'].mean()
        std_by_category = df.groupby('category')['amount'].std()
        
        for category in df['category'].unique():
            category_data = df[df['category'] == category]
            if len(category_data) > 0:
                mean = mean_by_category[category]
                std = std_by_category[category] if not pd.isna(std_by_category[category]) else 0
                
                unusual = category_data[
                    (category_data['amount'] < mean - 2*std) | 
                    (category_data['amount'] > mean + 2*std)
                ]
                
                if len(unusual) > 0:
                    insights.append(f"Unusual spending detected in {category}")
        
        # Spending trends
        monthly_spending = df.groupby(df['date'].dt.strftime('%Y-%m'))['amount'].sum()
        if len(monthly_spending) >= 2:
            current_month = monthly_spending.iloc[-1]
            previous_month = monthly_spending.iloc[-2]
            change = ((current_month - previous_month) / previous_month) * 100
            insights.append(f"Monthly spending changed by {change:.1f}% compared to last month")
        
        return insights

    def chat_response(self, query, transactions):
        if not transactions:
            return "I don't have any transaction data to analyze yet. Please add some transactions first!"

        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'])
        
        # Financial summary
        total_spent = abs(df[df['amount'] < 0]['amount'].sum())
        total_income = df[df['amount'] > 0]['amount'].sum()
        categories = df[df['amount'] < 0].groupby('category')['amount'].sum().abs()
        recent_transactions = df.sort_values('date', ascending=False).head(5)

        # Handle basic queries
        query = query.lower()
        
        # Basic responses without OpenAI
        if any(word in query for word in ['hi', 'hello', 'hey']):
            return f"Hello! Today's overview:\nTotal spent so far: ${total_spent:.2f}\nWould you like to know more about your spending?"

        try:
            # Enhanced OpenAI context
            system_prompt = """You are a sophisticated AI financial advisor. Analyze the data and provide:
            1. Clear, specific answers about transactions and spending
            2. Actionable financial insights
            3. Professional yet friendly responses
            4. Numbers formatted as currency with 2 decimal places
            5. Specific recommendations based on spending patterns"""

            context = f"""
            Financial Overview:
            - Total Income: ${total_income:.2f}
            - Total Expenses: ${total_spent:.2f}
            - Net Balance: ${total_income - total_spent:.2f}
            
            Category Breakdown:
            {categories.to_string()}
            
            Recent Transactions:
            {recent_transactions[['date', 'category', 'amount', 'description']].to_string()}
            
            User Question: {query}
            """

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.7,
                max_tokens=300,
                presence_penalty=0.6,
                frequency_penalty=0.3
            )
            return response.choices[0].message['content']
            
        except Exception as e:
            # Fallback to basic analysis if OpenAI fails
            if 'spent most' in query:
                return self._get_highest_spending(df)
            elif 'summary' in query:
                return self._get_summary(df)
            elif any(category in query for category in ['food', 'transport', 'entertainment', 'bills', 'other']):
                category = next(cat for cat in ['food', 'transport', 'entertainment', 'bills', 'other'] if cat in query)
                return self._get_category_analysis(category, df)
            
            # Date-specific queries
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', query)
            if date_match or 'today' in query or 'yesterday' in query:
                return self._get_date_spending(query, df)
            
            # Monthly analysis
            if any(word in query for word in ['month', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']):
                return self._get_monthly_spending(query, df)
            
            return "I apologize, but I encountered an error. You can ask me about:\n- Spending on specific dates\n- Monthly analysis\n- Category-wise spending\n- Overall summary"

    def _get_date_spending(self, query, df):
        if 'today' in query:
            date = datetime.now().strftime('%Y-%m-%d')
        elif 'yesterday' in query:
            date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', query)
            date = date_match.group(1)

        day_transactions = df[df['date'].dt.strftime('%Y-%m-%d') == date]
        if len(day_transactions) == 0:
            return f"No transactions found for {date}"

        total_spent = abs(day_transactions[day_transactions['amount'] < 0]['amount'].sum())
        response = f"On {date}, you spent ${total_spent:.2f}\n\nBreakdown:"
        
        for _, row in day_transactions.iterrows():
            if row['amount'] < 0:
                response += f"\n- ${abs(row['amount']):.2f} on {row['category']}: {row['description']}"
        
        return response

    def _get_monthly_spending(self, query, df):
        current_month = datetime.now().strftime('%Y-%m')
        if 'last month' in query:
            target_month = (datetime.now().replace(day=1) - timedelta(days=1)).strftime('%Y-%m')
        else:
            target_month = current_month

        month_data = df[df['date'].dt.strftime('%Y-%m') == target_month]
        if len(month_data) == 0:
            return f"No transactions found for {target_month}"

        total_spent = abs(month_data[month_data['amount'] < 0]['amount'].sum())
        category_spending = month_data[month_data['amount'] < 0].groupby('category')['amount'].sum().abs()

        response = f"Monthly Analysis for {target_month}:\nTotal spent: ${total_spent:.2f}\n\nCategory breakdown:"
        for category, amount in category_spending.items():
            response += f"\n- {category}: ${amount:.2f}"
        
        return response

    def _get_category_analysis(self, category, df):
        category_data = df[df['category'].str.lower() == category]
        if len(category_data) == 0:
            return f"No transactions found for category: {category}"

        total_spent = abs(category_data[category_data['amount'] < 0]['amount'].sum())
        recent_transactions = category_data.sort_values('date', ascending=False).head(3)

        response = f"Analysis for {category}:\nTotal spent: ${total_spent:.2f}\n\nRecent transactions:"
        for _, row in recent_transactions.iterrows():
            if row['amount'] < 0:
                response += f"\n- {row['date'].strftime('%Y-%m-%d')}: ${abs(row['amount']):.2f} - {row['description']}"
        
        return response

    def _get_highest_spending(self, df):
        category_spending = df[df['amount'] < 0].groupby('category')['amount'].sum().abs()
        highest_category = category_spending.idxmax()
        highest_amount = category_spending.max()
        
        return f"You spent most on {highest_category}: ${highest_amount:.2f}"

    def _get_summary(self, df):
        total_spent = abs(df[df['amount'] < 0]['amount'].sum())
        total_income = df[df['amount'] > 0]['amount'].sum()
        category_spending = df[df['amount'] < 0].groupby('category')['amount'].sum().abs()

        response = f"Financial Summary:\nTotal Income: ${total_income:.2f}\nTotal Expenses: ${total_spent:.2f}\n\nCategory-wise spending:"
        for category, amount in category_spending.items():
            response += f"\n- {category}: ${amount:.2f}"
        
        return response