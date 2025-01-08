# 🎈 Arbitrage Toolbox for Uniswap Liquidity Pools

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. (optional) Run unittests
   ```
   python -m unittest tests.test_app   
   ```

3. Run training code to update models
   ```
   $ python src/train.py
   ```

4. Run the basic streamlit app for exploring arbitrage basics

   ```
   $ streamlit run src/streamlit_arbitrage_playground.py
   ```

5. Run a streamlit app to provide a 1 minute forecast on arbitrage opportunities, and some analysis of the last 12 hours of transactions.
   ```
   $ streamlit run src/streamlit_arbitrage_app.py
   ```
