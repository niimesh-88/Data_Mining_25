pip install mlxtend
////////////////////
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# -----------------------------
# Dataset 1: Groceries Dataset
# -----------------------------
groceries = [['milk', 'bread'],
             ['bread', 'butter', 'jam'],
             ['milk', 'bread', 'butter'],
             ['bread'],
             ['milk', 'bread', 'butter', 'eggs']]

te = TransactionEncoder()
groceries_encoded = te.fit_transform(groceries)
df1 = pd.DataFrame(groceries_encoded, columns=te.columns_)

# Frequent Itemsets
frequent_items1 = apriori(df1, min_support=0.4, use_colnames=True)

# Association Rules
rules1 = association_rules(frequent_items1, metric="confidence", min_threshold=0.6)

print("Dataset 1: Groceries Frequent Itemsets\n", frequent_items1)
print("\nDataset 1: Association Rules\n", rules1)


# -----------------------------
# Dataset 2: Online Retail Dataset (Example)
# -----------------------------
retail = [['laptop', 'mouse'],
          ['mouse', 'keyboard'],
          ['laptop', 'mouse', 'keyboard'],
          ['mouse'],
          ['laptop', 'mouse', 'headphone']]

retail_encoded = te.fit_transform(retail)
df2 = pd.DataFrame(retail_encoded, columns=te.columns_)

# Frequent Itemsets
frequent_items2 = apriori(df2, min_support=0.4, use_colnames=True)

# Association Rules
rules2 = association_rules(frequent_items2, metric="lift", min_threshold=1.0)

print("\n\nDataset 2: Retail Frequent Itemsets\n", frequent_items2)
print("\nDataset 2: Association Rules\n", rules2)
