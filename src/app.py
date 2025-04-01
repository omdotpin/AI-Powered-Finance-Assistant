import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from finance_assistant import FinanceAssistant
from ai_helper import FinanceAI

def main():
    st.title("AI-Powered Personal Finance Assistant")
    
    finance = FinanceAssistant()
    ai_helper = FinanceAI()
    
    # Add sample data button in sidebar
    if st.sidebar.button("Load Sample Data"):
        load_sample_data(finance)
    
    # Navigation
    page = st.sidebar.selectbox("Navigation", 
        ["Dashboard", "Transactions", "Budget Management", "Analytics", "AI Insights", "Chat Assistant"])
    
    if page == "Chat Assistant":
        show_chat_assistant(finance, ai_helper)
    elif page == "AI Insights":
        show_ai_insights(finance, ai_helper)
    elif page == "Dashboard":
        show_dashboard(finance)
    elif page == "Transactions":
        show_transactions(finance)
    elif page == "Budget Management":
        show_budget_management(finance)
    elif page == "Analytics":
        show_analytics(finance)

def show_dashboard(finance):
    st.header("Financial Dashboard")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Income", f"${finance.get_total_income():.2f}")
    with col2:
        st.metric("Total Expenses", f"${abs(finance.get_total_expenses()):.2f}")
    with col3:
        balance = finance.get_total_income() + finance.get_total_expenses()
        st.metric("Net Balance", f"${balance:.2f}")
    
    # Budget Overview
    st.subheader("Budget Overview")
    budget_status = finance.get_budget_status()
    
    for category, status in budget_status.items():
        st.write(f"**{category}**")
        progress = status['percentage']
        st.progress(min(progress / 100, 1.0))
        st.write(f"Spent: ${status['spent']:.2f} / ${status['budget']:.2f}")

def show_transactions(finance):
    st.header("Manage Transactions")
    
    # Add transaction form
    with st.form("transaction_form"):
        amount = st.number_input("Amount", value=0.0)
        category = st.selectbox("Category", ["Income", "Food", "Transport", "Entertainment", "Bills", "Other"])
        description = st.text_input("Description")
        date = st.date_input("Date")
        
        if st.form_submit_button("Add Transaction"):
            finance.add_transaction(amount, category, description, date.strftime("%Y-%m-%d"))
            st.success("Transaction added successfully!")
    
    # Display transactions
    if finance.transactions:
        df = pd.DataFrame(finance.transactions)
        st.dataframe(df)

def show_budget_management(finance):
    st.header("Budget Management")
    
    # Set budget form
    with st.form("budget_form"):
        category = st.selectbox("Category", ["Food", "Transport", "Entertainment", "Bills", "Other"])
        budget_amount = st.number_input("Monthly Budget Amount", min_value=0.0)
        
        if st.form_submit_button("Set Budget"):
            finance.set_budget(category, budget_amount)
            st.success(f"Budget for {category} set to ${budget_amount:.2f}")
    
    # Display current budgets
    st.subheader("Current Budgets")
    budget_status = finance.get_budget_status()
    for category, status in budget_status.items():
        st.write(f"**{category}**: ${status['budget']:.2f}")

def show_analytics(finance):
    st.header("Financial Analytics")
    
    # Spending trends
    st.subheader("Spending Trends")
    trends_df = finance.get_spending_trends()
    if not trends_df.empty:
        fig = px.line(trends_df, x='date', y='amount', color='category',
                      title='Daily Spending by Category')
        st.plotly_chart(fig)
    
    # Category distribution
    if finance.transactions:
        df = pd.DataFrame(finance.transactions)
        expenses_by_category = df[df['amount'] < 0].groupby('category')['amount'].sum().abs()
        fig = px.pie(values=expenses_by_category.values, names=expenses_by_category.index,
                     title='Expense Distribution by Category')
        st.plotly_chart(fig)

def show_ai_insights(finance, ai_helper):
    st.header("AI-Powered Financial Insights")
    
    # Prepare data for AI analysis
    if finance.transactions:
        df = ai_helper.prepare_data(finance.transactions)
        
        # Expense Predictions
        st.subheader("Expense Predictions")
        predictions = ai_helper.predict_expenses(df)
        if predictions:
            fig = go.Figure()
            categories = list(predictions.keys())
            values = list(predictions.values())
            
            fig.add_trace(go.Bar(x=categories, y=values))
            fig.update_layout(title="Predicted Monthly Expenses by Category")
            st.plotly_chart(fig)
        
        # Smart Insights
        st.subheader("Smart Insights")
        insights = ai_helper.get_insights(finance.transactions)
        for insight in insights:
            st.info(insight)
        
        # Spending Patterns
        st.subheader("Spending Pattern Analysis")
        if not df.empty:
            daily_patterns = df.groupby('day_of_week')['amount'].mean()
            fig = px.line(x=daily_patterns.index, y=daily_patterns.values,
                         labels={'x': 'Day of Week', 'y': 'Average Spending'},
                         title='Spending Patterns by Day of Week')
            st.plotly_chart(fig)
    else:
        st.warning("Add some transactions to get AI-powered insights!")

def show_chat_assistant(finance, ai_helper):
    st.header("💬 AI Financial Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        welcome_msg = "Hello! I'm your AI financial assistant. I can help you analyze your spending and answer questions about your finances. What would you like to know?"
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your finances..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            response = ai_helper.chat_response(prompt, finance.transactions)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

def load_sample_data(finance):
    # Sample transactions
    sample_data = [
        {"amount": 5000.00, "category": "Income", "description": "Monthly Salary", "date": "2024-03-01"},
        {"amount": 1000.00, "category": "Income", "description": "Freelance Work", "date": "2024-03-15"},
        {"amount": -50.00, "category": "Food", "description": "Grocery Shopping", "date": "2024-03-02"},
        {"amount": -30.00, "category": "Food", "description": "Restaurant", "date": "2024-03-05"},
        {"amount": -100.00, "category": "Transport", "description": "Monthly Bus Pass", "date": "2024-03-01"},
        {"amount": -80.00, "category": "Entertainment", "description": "Movie Night", "date": "2024-03-09"},
        {"amount": -200.00, "category": "Bills", "description": "Electricity Bill", "date": "2024-03-05"},
        {"amount": -800.00, "category": "Bills", "description": "Rent", "date": "2024-03-01"},
        {"amount": -60.00, "category": "Other", "description": "Gift", "date": "2024-03-12"}
    ]
    
    # Sample budgets
    sample_budgets = {
        "Food": 500.00,
        "Transport": 200.00,
        "Entertainment": 300.00,
        "Bills": 1200.00,
        "Other": 200.00
    }
    
    # Load transactions
    for transaction in sample_data:
        finance.add_transaction(
            transaction["amount"],
            transaction["category"],
            transaction["description"],
            transaction["date"]
        )
    
    # Set budgets
    for category, amount in sample_budgets.items():
        finance.set_budget(category, amount)
    
    st.sidebar.success("Sample data loaded successfully!")

if __name__ == "__main__":
    main()