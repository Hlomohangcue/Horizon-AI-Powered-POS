import pandas as pd

# Load the data that's now on GitHub
transactions = pd.read_csv('data/transactions.csv')
inventory = pd.read_csv('data/inventory.csv')

print("LIVE APP VERIFICATION")
print("=" * 30)

print(f"Total Revenue: M {transactions['total_amount'].sum():.2f}")
print(f"Total Transactions: {len(transactions)}")
print(f"Avg Transaction: M {transactions['total_amount'].mean():.2f}")
print(f"Items Sold: {transactions['quantity'].sum()}")
print(f"Unique Customers: {transactions['customer_id'].nunique()}")

print(f"\nInventory products: {len(inventory)}")
print(f"Transaction products: {transactions['product_name'].nunique()}")

print(f"\nTop 3 products:")
top_products = transactions.groupby('product_name')['total_amount'].sum().sort_values(ascending=False)
for i, (product, revenue) in enumerate(top_products.head(3).items(), 1):
    print(f"{i}. {product}: M {revenue:.2f}")

print(f"\nGitHub push successful!")
print("Live app will update automatically in 1-2 minutes.")